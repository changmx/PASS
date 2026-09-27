"""Relativistic 2D3V electron motion in prescribed beam and magnetic fields."""

import numpy as np

from PASS.utils.constants import const


def electron_gamma(ux, uy, uz, xp=np):
    return xp.hypot(1, xp.hypot(xp.hypot(ux, uy), uz))


def electron_energy_ev(ux, uy, uz, xp=np):
    """Kinetic energy with a cancellation-free nonrelativistic limit."""
    normalized_magnitude = xp.hypot(xp.hypot(ux, uy), uz)
    return const.m_e_eV * (normalized_magnitude / (xp.hypot(1, normalized_magnitude) + 1)) * normalized_magnitude


def normalized_momentum(energy_ev, xp=np):
    """Return the dimensionless mechanical momentum magnitude P/(m_e*c)."""
    kinetic_ratio = xp.asarray(energy_ev) / const.m_e_eV
    return xp.sqrt(kinetic_ratio * (kinetic_ratio + 2))


def round_gaussian_beam_field(x, y, line_charge, beam_sigma, chamber_radius, xp=np):
    """E in V/m for an on-axis round Gaussian truncated at the round wall.

    The signed line charge is the total charge per metre inside the chamber.
    Axial symmetry makes the grounded circular boundary change only Phi's gauge.
    """
    inverse_width_squared = 1 / (2 * beam_sigma**2)
    normalization = -np.expm1(-chamber_radius**2 * inverse_width_squared)
    radius_squared = x * x + y * y
    fraction = -xp.expm1(-radius_squared * inverse_width_squared)
    radial_factor = xp.where(radius_squared > 0, fraction / xp.where(radius_squared > 0, radius_squared, 1), inverse_width_squared)
    scale = line_charge / (2 * const.pi * const.epsilon0 * normalization)
    return scale * radial_factor * x, scale * radial_factor * y


def boris_push(ux, uy, uz, ex, ey, bx, by, bz, dt, xp=np):
    """Advance dimensionless u=P/(m_e*c), with physical electron charge -e."""
    half_electric = -const.e * dt / (2 * const.m_e_kg * const.c)
    half_magnetic = -const.e * dt / (2 * const.m_e_kg)
    minus_x, minus_y, minus_z = ux + half_electric * ex, uy + half_electric * ey, uz
    gamma = electron_gamma(minus_x, minus_y, minus_z, xp)
    tx, ty, tz = half_magnetic * bx / gamma, half_magnetic * by / gamma, half_magnetic * bz / gamma
    prime_x = minus_x + minus_y * tz - minus_z * ty
    prime_y = minus_y + minus_z * tx - minus_x * tz
    prime_z = minus_z + minus_x * ty - minus_y * tx
    scale = 2 / (1 + tx * tx + ty * ty + tz * tz)
    plus_x = minus_x + scale * (prime_y * tz - prime_z * ty)
    plus_y = minus_y + scale * (prime_z * tx - prime_x * tz)
    plus_z = minus_z + scale * (prime_x * ty - prime_y * tx)
    return plus_x + half_electric * ex, plus_y + half_electric * ey, plus_z


def external_magnetic_field(x, y, magnetic_field, magnetic_gradient=0.):
    """Normal quadrupole B=(B0x+G*y, B0y+G*x, B0z), in T with G in T/m.

    The caller supplies the particle positions at the kick midpoint. Keeping
    the zero-gradient branch scalar preserves the uniform-field arithmetic.
    """
    bx, by, bz = magnetic_field
    if magnetic_gradient != 0:
        bx = bx + magnetic_gradient * y
        by = by + magnetic_gradient * x
    return bx, by, bz


def first_circle_hit(x, y, vx, vy, radius, xp=np):
    """First forward intersection time for a straight drift from inside a circle."""
    speed = xp.hypot(vx, vy)
    safe_speed = xp.where(speed > 0, speed, 1)
    scaled_x, scaled_y = x / radius, y / radius
    scaled_radius = xp.hypot(scaled_x, scaled_y)
    radial_velocity = scaled_x * (vx / safe_speed) + scaled_y * (vy / safe_speed)
    distance_squared = xp.maximum((1 - scaled_radius) * (1 + scaled_radius), 0)
    # Scale the quadratic before solving so a finite large chamber cannot
    # overflow the discriminant through a factor of radius squared times c squared.
    discriminant = xp.hypot(radial_velocity, xp.sqrt(distance_squared))
    # Rationalize the outgoing root near the wall to avoid cancellation.
    outgoing = distance_squared / xp.where(discriminant + radial_velocity > 0, discriminant + radial_velocity, 1)
    inward = discriminant - radial_velocity
    distance = radius * xp.where(radial_velocity >= 0, outgoing, inward)
    return xp.where(speed > 0, distance / safe_speed, xp.inf)


class _GpuElectronPusher:
    """Sequential GPU workspace; returned arrays borrow reusable storage.

    Particle vectors are contiguous float64 arrays on the creation device and
    stream. Kick/statistic results share one buffer. Drift remainders use a
    separate buffer that survives kick/statistic calls until the next drift.
    No method consumes random numbers or changes particle membership.
    """

    def __init__(self):
        import cupy as cp

        self.xp = cp
        self.device = cp.cuda.runtime.getDevice()
        self.stream = cp.cuda.get_current_stream()
        self._initialize_gpu_kernels()
        self._scratch = self._remaining = None
        self._wall_mask = self._wall_diagnostics = None
        self._status = cp.empty(1, dtype=cp.int32)
        self._closed = False

    def _vectors(self, arrays, names):
        cp = self.xp
        if self._closed:
            raise RuntimeError("electron pusher resources are closed")
        if cp.cuda.runtime.getDevice() != self.device or cp.cuda.get_current_stream().ptr != self.stream.ptr:
            raise RuntimeError("electron pusher requires its creation device and stream")
        values = tuple(arrays[name] for name in names)
        shape = values[0].shape if isinstance(values[0], cp.ndarray) else None
        for name, value in zip(names, values):
            self._check_vector(value, shape, name)
        return values

    def _check_vector(self, value, shape, name):
        if (not isinstance(value, self.xp.ndarray) or value.dtype != np.float64 or value.ndim != 1 or value.shape != shape
                or not value.flags.c_contiguous or value.device.id != self.device):
            raise TypeError(f"electron pusher {name} must be a matching contiguous float64 GPU vector")

    @staticmethod
    def _scalar(value, name, *, nonnegative=False):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"electron pusher {name} must be a finite real scalar")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"electron pusher {name} must be finite") from exc
        if not np.isfinite(value) or nonnegative and value < 0:
            raise ValueError(f"electron pusher {name} must be finite" + (" and nonnegative" if nonnegative else ""))
        return np.float64(value)

    def _buffer(self, name, count, dtype=None):
        buffer = getattr(self, name)
        if buffer is None or buffer.size < count:
            capacity = count if buffer is None else max(count, 2 * buffer.size)
            buffer = self.xp.empty(capacity, dtype=self.xp.float64 if dtype is None else dtype)
            setattr(self, name, buffer)
        return buffer[:count]

    def _fields(self, values, fallback):
        pointers, scalars, mask = [], [], 0
        for index, value in enumerate(values):
            if isinstance(value, self.xp.ndarray):
                self._check_vector(value, fallback.shape, "field")
                pointers.append(value)
                scalars.append(np.float64(0))
                mask |= 1 << index
            else:
                pointers.append(fallback)
                scalars.append(self._scalar(value, "field"))
        return (*pointers, *scalars, np.int32(mask))

    @staticmethod
    def _launch(kernel, count, arguments):
        if count:
            kernel(((count + 255) // 256, ), (256, ), arguments)

    def kick(self, arrays, cloud_ex, cloud_ey, beam_ex, beam_ey, magnetic_field, beam_beta, dt, magnetic_gradient=0.):
        """Update u in place and return weighted kinetic-energy changes in eV.

        Each E input is a GPU vector or a finite host scalar in V/m. The
        external magnetic field is a host three-vector in Tesla, with optional
        normal quadrupole gradient in T/m sampled at arrays' midpoint x/y.
        Beam B is beta/c times z-hat cross beam E. The caller owns reduction.
        """
        ux, uy, uz, weight = self._vectors(arrays, ("ux", "uy", "uz", "weight"))
        fields = self._fields((cloud_ex, cloud_ey, beam_ex, beam_ey), ux)
        magnetic_field = np.asarray(magnetic_field)
        if magnetic_field.shape != (3, ) or magnetic_field.dtype.kind not in "fiu":
            raise ValueError("electron pusher magnetic_field must contain three finite Tesla values")
        magnetic_field = tuple(self._scalar(value, "magnetic field") for value in magnetic_field)
        magnetic_gradient = self._scalar(magnetic_gradient, "magnetic_gradient")
        x, y = ux, ux
        if magnetic_gradient != 0:
            x, y = self._vectors(arrays, ("x", "y"))
            self._check_vector(x, ux.shape, "x")
            self._check_vector(y, ux.shape, "y")
        beam_beta = self._scalar(beam_beta, "beam_beta", nonnegative=True)
        if beam_beta > 1:
            raise ValueError("electron pusher beam_beta must be in [0,1]")
        dt = self._scalar(dt, "dt", nonnegative=True)
        output = self._buffer("_scratch", ux.size)
        half_electric = np.float64(-const.e * dt / (2 * const.m_e_kg * const.c))
        half_magnetic = np.float64(-const.e * dt / (2 * const.m_e_kg))
        self._launch(self._kick_kernel, ux.size,
                     (ux, uy, uz, weight, output, *fields, *magnetic_field, x, y, magnetic_gradient, beam_beta, half_electric, half_magnetic,
                      np.float64(const.c), np.float64(const.m_e_eV), np.int64(ux.size)))
        return output

    def transverse_speed(self, arrays):
        """Return each particle's transverse speed in m/s in borrowed storage."""
        return self._quantity(arrays, 0)

    def weighted_energy(self, arrays):
        """Return each particle's weight times kinetic energy in eV."""
        return self._quantity(arrays, 1)

    def _quantity(self, arrays, mode):
        ux, uy, uz, weight = self._vectors(arrays, ("ux", "uy", "uz", "weight"))
        output = self._buffer("_scratch", ux.size)
        self._launch(self._quantity_kernel, ux.size,
                     (ux, uy, uz, weight, output, np.int32(mode), np.float64(const.c), np.float64(const.m_e_eV), np.int64(ux.size)))
        return output

    def prepare_drift(self, arrays, duration, radius):
        """Advance only wall-free particles; retain hitting particles for the wall loop.

        The returned remaining times are zero for completed free drifts and
        ``duration`` for wall candidates. Wall candidates retain their original
        coordinates and indices, preserving the caller's random-draw order.
        """
        x, y, ux, uy, uz = self._vectors(arrays, ("x", "y", "ux", "uy", "uz"))
        duration = self._scalar(duration, "duration", nonnegative=True)
        radius = self._scalar(radius, "radius")
        if radius <= 0:
            raise ValueError("electron pusher radius must be positive")
        output = self._buffer("_remaining", x.size)
        self._launch(self._drift_kernel, x.size, (x, y, ux, uy, uz, output, duration, radius, np.float64(const.c), np.int64(x.size)))
        return output

    def close(self):
        """Drop workspace buffers; no physical particle state is owned here."""
        self._scratch = self._remaining = None
        self._wall_mask = self._wall_diagnostics = None
        self._status = None
        self._closed = True

    def drift_wall(self, arrays, remaining, hits, radius, max_hits):
        """Advance one collision round and return ordered impacts plus a zero flag.

        Every particle advances to its next impact or completes its remaining
        drift. Integer flags may be unordered; particle indices never are.
        """
        values = self._vectors(arrays, ("x", "y", "ux", "uy", "uz", "weight"))
        self._check_vector(remaining, values[0].shape, "remaining")
        if hits.dtype != self.xp.int64 or hits.shape != remaining.shape or not hits.flags.c_contiguous:
            raise TypeError("electron pusher wall hit counts must be a matching contiguous int64 GPU vector")
        radius = self._scalar(radius, "radius")
        if radius <= 0 or isinstance(max_hits, (bool, np.bool_)) or not isinstance(max_hits, (int, np.integer)) or max_hits < 1:
            raise ValueError("electron pusher wall radius and hit limit must be positive")
        count = remaining.size
        if not count:
            return None, False
        mask = self._buffer("_wall_mask", count, dtype=self.xp.bool_)
        self._status.fill(0)
        self._launch(self._wall_drift_kernel, count,
                     (*values, remaining, hits, mask, self._status, radius, np.int64(max_hits), np.float64(const.c), np.int64(count)))
        status = int(self._status.get()[0])
        if status & 4:
            raise ValueError("electron wall events exceed max_wall_hits_per_step; reduce max_time_step")
        return self.xp.flatnonzero(mask) if status & 1 else None, bool(status & 2)

    def emit_wall(self, arrays, remaining, impacted, mu, azimuth, radius, peak_yield, peak_energy_ev, shape, emission_energy_ev):
        """Apply one ordered emission batch; return four borrowed ledger vectors.

        The host supplies exactly the legacy PCG64 draws, including absorbed
        impacts. Each output row is independently reduced by the caller.
        """
        values = self._vectors(arrays, ("x", "y", "ux", "uy", "uz", "weight"))
        self._check_vector(remaining, values[0].shape, "remaining")
        count = impacted.size
        random_values = self.xp.asarray(np.stack((mu, azimuth)))
        output = self._buffer("_wall_diagnostics", 4 * count).reshape(4, count)
        self._launch(self._wall_emission_kernel, count,
                     (*values, remaining, impacted, random_values, output, np.float64(radius * (1 - 64 * np.finfo(float).eps)),
                      np.float64(peak_yield), np.float64(np.log(peak_energy_ev)), np.float64(shape), np.float64(
                          np.log(shape)), np.float64(np.log(shape - 1)), np.float64(emission_energy_ev), np.float64(const.m_e_eV), np.int64(count)))
        return output

    def validate(self, arrays, radius):
        """Check finite state, nonnegative weights and the circular wall in one sync."""
        values = self._vectors(arrays, ("x", "y", "ux", "uy", "uz", "weight"))
        radius = self._scalar(radius, "radius")
        if radius <= 0:
            raise ValueError("electron pusher radius must be positive")
        count = values[0].size
        if not count:
            return
        self._status.fill(0)
        self._launch(self._validate_kernel, count, (*values, self._status, radius, np.int64(count)))
        status = int(self._status.get()[0])
        if status & 1:
            raise ValueError("electron cloud integration produced nonfinite particle state (GPU status 1)")
        if status & 2:
            raise ValueError("electron cloud weights must be nonnegative (GPU status 2)")
        if status & 4:
            raise ValueError("dynamic electrons must lie strictly inside the round chamber (GPU status 4)")

    def _initialize_gpu_kernels(self):
        self._module = self.xp.RawModule(code=r"""
__device__ double electron_gamma_value(
    double ux,
    double uy,
    double uz
) {
    return hypot(1.0, hypot(hypot(ux, uy), uz));
}

__device__ double electron_energy_value(
    double ux,
    double uy,
    double uz,
    double mass_energy_ev
) {
    double magnitude = hypot(hypot(ux, uy), uz);
    return mass_energy_ev * (magnitude / (hypot(1.0, magnitude) + 1.0)) * magnitude;
}

extern "C" __global__ void push_electrons(
    double* ux,
    double* uy,
    double* uz,
    const double* weight,
    double* energy_delta,
    const double* cloud_ex,
    const double* cloud_ey,
    const double* beam_ex,
    const double* beam_ey,
    double cloud_ex_scalar,
    double cloud_ey_scalar,
    double beam_ex_scalar,
    double beam_ey_scalar,
    int field_mask,
    double external_bx,
    double external_by,
    double external_bz,
    const double* x,
    const double* y,
    double magnetic_gradient,
    double beam_beta,
    double half_electric,
    double half_magnetic,
    double speed_of_light,
    double mass_energy_ev,
    long long count
) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    double source_ex = (field_mask & 4) ? beam_ex[index] : beam_ex_scalar;
    double source_ey = (field_mask & 8) ? beam_ey[index] : beam_ey_scalar;
    double ex = ((field_mask & 1) ? cloud_ex[index] : cloud_ex_scalar) + source_ex;
    double ey = ((field_mask & 2) ? cloud_ey[index] : cloud_ey_scalar) + source_ey;
    if (magnetic_gradient != 0.0) {
        external_bx += magnetic_gradient * y[index];
        external_by += magnetic_gradient * x[index];
    }
    double bx = external_bx - beam_beta * source_ey / speed_of_light;
    double by = external_by + beam_beta * source_ex / speed_of_light;
    double original_x = ux[index];
    double original_y = uy[index];
    double original_z = uz[index];
    double energy_before = electron_energy_value(original_x, original_y, original_z, mass_energy_ev);
    double minus_x = original_x + half_electric * ex;
    double minus_y = original_y + half_electric * ey;
    double minus_z = original_z;
    double gamma = electron_gamma_value(minus_x, minus_y, minus_z);
    double tx = half_magnetic * bx / gamma;
    double ty = half_magnetic * by / gamma;
    double tz = half_magnetic * external_bz / gamma;
    double prime_x = minus_x + minus_y * tz - minus_z * ty;
    double prime_y = minus_y + minus_z * tx - minus_x * tz;
    double prime_z = minus_z + minus_x * ty - minus_y * tx;
    double scale = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
    double plus_x = minus_x + scale * (prime_y * tz - prime_z * ty);
    double plus_y = minus_y + scale * (prime_z * tx - prime_x * tz);
    double plus_z = minus_z + scale * (prime_x * ty - prime_y * tx);
    double next_x = plus_x + half_electric * ex;
    double next_y = plus_y + half_electric * ey;
    ux[index] = next_x;
    uy[index] = next_y;
    uz[index] = plus_z;
    double energy_after = electron_energy_value(next_x, next_y, plus_z, mass_energy_ev);
    energy_delta[index] = weight[index] * (energy_after - energy_before);
}

extern "C" __global__ void electron_quantity(
    const double* ux,
    const double* uy,
    const double* uz,
    const double* weight,
    double* output,
    int mode,
    double speed_of_light,
    double mass_energy_ev,
    long long count
) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    double x = ux[index];
    double y = uy[index];
    double z = uz[index];
    output[index] =
        mode == 0 ? speed_of_light * (hypot(x, y) / electron_gamma_value(x, y, z)) : weight[index] * electron_energy_value(x, y, z, mass_energy_ev);
}

extern "C" __global__ void prepare_electron_drift(
    double* x,
    double* y,
    const double* ux,
    const double* uy,
    const double* uz,
    double* remaining,
    double duration,
    double radius,
    double speed_of_light,
    long long count
) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    double gamma = electron_gamma_value(ux[index], uy[index], uz[index]);
    double vx = speed_of_light * (ux[index] / gamma);
    double vy = speed_of_light * (uy[index] / gamma);
    double speed = hypot(vx, vy);
    double safe_speed = speed > 0.0 ? speed : 1.0;
    double scaled_x = x[index] / radius;
    double scaled_y = y[index] / radius;
    double scaled_radius = hypot(scaled_x, scaled_y);
    double radial_velocity = scaled_x * (vx / safe_speed) + scaled_y * (vy / safe_speed);
    double distance_squared = (1.0 - scaled_radius) * (1.0 + scaled_radius);
    if (distance_squared < 0.0) {
        distance_squared = 0.0;
    }
    double discriminant = hypot(radial_velocity, sqrt(distance_squared));
    double denominator = discriminant + radial_velocity;
    double outgoing = distance_squared / (denominator > 0.0 ? denominator : 1.0);
    double inward = discriminant - radial_velocity;
    double distance = radius * (radial_velocity >= 0.0 ? outgoing : inward);
    double hit_time = speed > 0.0 ? distance / safe_speed : __longlong_as_double(0x7ff0000000000000LL);
    if (hit_time <= duration) {
        remaining[index] = duration;
    } else {
        x[index] += vx * duration;
        y[index] += vy * duration;
        remaining[index] = 0.0;
    }
}

extern "C" __global__ void drift_wall_round(
    double* x,
    double* y,
    const double* ux,
    const double* uy,
    const double* uz,
    const double* weight,
    double* remaining,
    long long* hits,
    bool* impacted,
    int* status,
    double radius,
    long long max_hits,
    double speed_of_light,
    long long count
) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    impacted[index] = false;
    if (weight[index] == 0.0) {
        atomicOr(status, 2);
    }
    double duration = remaining[index];
    if (!(duration > 0.0)) {
        return;
    }
    double gamma = electron_gamma_value(ux[index], uy[index], uz[index]);
    double vx = speed_of_light * (ux[index] / gamma);
    double vy = speed_of_light * (uy[index] / gamma);
    double speed = hypot(vx, vy);
    double safe_speed = speed > 0.0 ? speed : 1.0;
    double scaled_x = x[index] / radius;
    double scaled_y = y[index] / radius;
    double scaled_radius = hypot(scaled_x, scaled_y);
    double radial_velocity = scaled_x * (vx / safe_speed) + scaled_y * (vy / safe_speed);
    double distance_squared = (1.0 - scaled_radius) * (1.0 + scaled_radius);
    if (distance_squared < 0.0) {
        distance_squared = 0.0;
    }
    double discriminant = hypot(radial_velocity, sqrt(distance_squared));
    double denominator = discriminant + radial_velocity;
    double outgoing = distance_squared / (denominator > 0.0 ? denominator : 1.0);
    double inward = discriminant - radial_velocity;
    double distance = radius * (radial_velocity >= 0.0 ? outgoing : inward);
    double hit_time = speed > 0.0 ? distance / safe_speed : __longlong_as_double(0x7ff0000000000000LL);
    bool crossing = hit_time <= duration;
    double elapsed = crossing ? hit_time : duration;
    x[index] = x[index] + vx * elapsed;
    y[index] = y[index] + vy * elapsed;
    remaining[index] = crossing ? duration - elapsed : 0.0;
    if (crossing) {
        impacted[index] = true;
        hits[index] += 1;
        atomicOr(status, hits[index] > max_hits ? 5 : 1);
    }
}

extern "C" __global__ void emit_wall_electrons(
    double* x,
    double* y,
    double* ux,
    double* uy,
    double* uz,
    double* weight,
    double* remaining,
    const long long* impacted,
    const double* random_values,
    double* diagnostics,
    double radius_inside,
    double peak_yield,
    double log_peak_energy,
    double shape,
    double log_shape,
    double log_shape_minus_one,
    double emission_energy_ev,
    double mass_energy_ev,
    long long count
) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    long long particle = impacted[index];
    double incident_energy = electron_energy_value(ux[particle], uy[particle], uz[particle], mass_energy_ev);
    double incident_weight = weight[particle];
    double log_ratio = log(incident_energy > 0.0 ? incident_energy : 1.0) - log_peak_energy;
    double scaled_log_ratio = shape * log_ratio;
    // Match CuPy logaddexp, including its equal-argument branch.
    double log_denominator = log_shape_minus_one == scaled_log_ratio
                                 ? log_shape_minus_one + log(2.0)
                                 : fmax(log_shape_minus_one, scaled_log_ratio) + log1p(exp(-fabs(log_shape_minus_one - scaled_log_ratio)));
    double logarithmic_yield = log_shape + log_ratio - log_denominator;
    double raw_yield = incident_energy > 0.0 ? peak_yield * exp(logarithmic_yield) : 0.0;
    double energy_cap = incident_energy / emission_energy_ev;
    double emitted_yield = raw_yield < energy_cap ? raw_yield : energy_cap;
    double emitted_weight = incident_weight * emitted_yield;
    double emitted_energy = incident_energy < emission_energy_ev ? incident_energy : emission_energy_ev;
    diagnostics[index] = incident_weight;
    diagnostics[count + index] = emitted_weight;
    diagnostics[2 * count + index] = incident_weight * incident_energy;
    diagnostics[3 * count + index] = emitted_weight * emitted_energy;
    double normal_radius = hypot(x[particle], y[particle]);
    double nx = -x[particle] / normal_radius;
    double ny = -y[particle] / normal_radius;
    double mu = random_values[index];
    double azimuth = random_values[count + index];
    double tangential_squared = 1.0 - mu * mu;
    double tangential = sqrt(tangential_squared > 0.0 ? tangential_squared : 0.0);
    double kinetic_ratio = emitted_energy / mass_energy_ev;
    double magnitude = sqrt(kinetic_ratio * (kinetic_ratio + 2.0));
    double tangent_x = -ny;
    ux[particle] = magnitude * (mu * nx + tangential * cos(azimuth) * tangent_x);
    uy[particle] = magnitude * (mu * ny + tangential * cos(azimuth) * nx);
    uz[particle] = magnitude * tangential * sin(azimuth);
    x[particle] = -radius_inside * nx;
    y[particle] = -radius_inside * ny;
    weight[particle] = emitted_weight;
    if (emitted_weight == 0.0) {
        remaining[particle] = 0.0;
    }
}

extern "C" __global__ void validate_electrons(
    const double* x,
    const double* y,
    const double* ux,
    const double* uy,
    const double* uz,
    const double* weight,
    int* status,
    double radius,
    long long count
) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    int flags = 0;
    if (!isfinite(x[index]) || !isfinite(y[index]) || !isfinite(ux[index]) || !isfinite(uy[index]) || !isfinite(uz[index]) ||
        !isfinite(weight[index])) {
        flags |= 1;
    }
    if (weight[index] < 0.0) {
        flags |= 2;
    }
    if (!(hypot(x[index], y[index]) < radius)) {
        flags |= 4;
    }
    if (flags) {
        atomicOr(status, flags);
    }
}
""",
                                         options=("--std=c++17", "--fmad=false"))
        self._kick_kernel = self._module.get_function("push_electrons")
        self._quantity_kernel = self._module.get_function("electron_quantity")
        self._drift_kernel = self._module.get_function("prepare_electron_drift")
        self._wall_drift_kernel = self._module.get_function("drift_wall_round")
        self._wall_emission_kernel = self._module.get_function("emit_wall_electrons")
        self._validate_kernel = self._module.get_function("validate_electrons")
