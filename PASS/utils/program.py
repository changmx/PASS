"""Prescribed piecewise-linear time programs for CPU and GPU consumers."""

from decimal import Decimal, localcontext

import numpy as np


class LinearProgram:
    """Linear samples with constant endpoint extrapolation and exact integral.

    Times are physical seconds. Phase samples must already be unwrapped.
    Evaluation uses a scalar reference time plus a local particle time difference.
    Sample arrays are owned, read-only snapshots so cached device tables stay valid.
    """

    def __init__(self, values, times=None, *, origin=0.0):
        self.origin = float(origin)
        self.values = np.atleast_1d(np.array(values, dtype=float, copy=True))
        self.times = np.array([origin] if times is None else times, dtype=float, copy=True)
        if (self.times.ndim != 1 or self.values.ndim != 1 or not len(self.times) or not len(self.values) or not np.all(np.isfinite(self.times))
                or not np.all(np.isfinite(self.values)) or not np.isfinite(self.origin) or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Programs require finite values and strictly increasing times")
        if len(self.values) == 1:
            self.times = np.asarray([origin], dtype=float)
        elif len(self.values) != len(self.times):
            raise ValueError("Program values and times must have equal lengths")
        self.slopes = np.r_[np.diff(self.values) / np.diff(self.times), 0.] if len(self.times) > 1 else np.zeros(1)
        self.integrals = np.r_[0., np.cumsum(np.diff(self.times) * (self.values[1:] + self.values[:-1]) / 2)]
        self._devices = {}
        self._phase_data = None
        self._phase_anchor = None
        self._phase_devices = {}
        self.integral_origin = float(self._primitive(origin, 0., np))
        for array in (self.times, self.values, self.slopes, self.integrals):
            array.setflags(write=False)

    def _arrays(self, xp):
        if xp not in self._devices:
            self._devices[xp] = tuple(xp.asarray(a) for a in (self.times, self.values, self.slopes, self.integrals))
        return self._devices[xp]

    def _parts(self, reference, offset, xp):
        if xp is np and np.isscalar(offset):
            index = (0 if len(self.times) == 1 else max(0, int(np.searchsorted(self.times - reference, offset, side='right')) - 1))
            dx = (reference - self.times[index]) + offset
            slope = 0. if (reference - self.times[0]) + offset < 0. else self.slopes[index]
            return self.values[index], slope, dx, self.integrals[index]
        times, values, slopes, integrals = self._arrays(xp)
        index = xp.clip(xp.searchsorted(times - reference, offset, side='right') - 1, 0, len(self.times) - 1)
        dx = (reference - times[index]) + offset
        slope = xp.where((reference - self.times[0]) + offset < 0, 0., slopes[index])
        return values[index], slope, dx, integrals[index]

    def value(self, reference, offset=0., xp=np):
        if xp is np and np.isscalar(offset) and len(self.values) == 1:
            return self.values[0]
        value, slope, dx, _ = self._parts(reference, offset, xp)
        return value + slope * dx

    def _primitive(self, reference, offset, xp):
        value, slope, dx, base = self._parts(reference, offset, xp)
        return base + dx * (value + .5 * slope * dx)

    def integral(self, reference, offset=0., xp=np):
        return self._primitive(reference, offset, xp) - self.integral_origin

    def phase_cycles(self, reference, offset=0., xp=np):
        # The epoch product itself must be reduced before rounding to float64.
        # Merely taking remainder(f*t, 1) loses phase at large elapsed times.
        base, frequency, reference_index = self.phase_anchor(reference)
        if len(self.values) == 1:
            return xp.remainder(base + frequency * offset, 1.)
        if xp is np and np.isscalar(offset) and offset == 0.:
            return base
        times, values, slopes, _ = self._arrays(xp)
        index = xp.clip(xp.searchsorted(times - reference, offset, side='right') - 1, 0, len(self.times) - 1)
        before = offset < times[0] - reference
        slope = xp.where(before, 0., slopes[index])
        same = (index == reference_index) & (before == (reference < self.times[0]))
        local = base + offset * (frequency + .5 * slope * offset)
        if xp not in self._phase_devices:
            self._phase_devices[xp] = xp.asarray(self.phase_integrals)
        following = xp.minimum(index + 1, len(self.times) - 1)
        use_upper = (~before & (following != index) & (xp.abs(reference - times[following]) < xp.abs(reference - times[index])))
        anchor = xp.where(use_upper, following, index)
        dx = (reference - times[anchor]) + offset
        crossing = self._phase_devices[xp][anchor] + dx * (values[anchor] + .5 * slope * dx)
        return xp.remainder(xp.where(same, local, crossing), 1.)

    def _prepare_phase_data(self):
        """Exact decimal input products, built lazily for frequency consumers."""
        if self._phase_data is not None:
            return
        with localcontext() as ctx:
            ctx.prec = 80
            times = tuple(Decimal.from_float(float(t)) for t in self.times)
            values = tuple(Decimal.from_float(float(v)) for v in self.values)
            integrals = [Decimal(0)]
            for j in range(len(times) - 1):
                integrals.append(integrals[-1] + (times[j + 1] - times[j]) * (values[j + 1] + values[j]) / 2)
            self._phase_data = times, values, tuple(integrals)
            origin = self._decimal_primitive(self.origin)
            self._phase_origin = origin
            self._phase_integrals = np.array([float((v - origin) % 1) % 1. for v in integrals])
            self._phase_integrals.setflags(write=False)

    def _decimal_primitive(self, reference):
        times, values, integrals = self._phase_data
        j = max(0, int(np.searchsorted(self.times, reference, side='right')) - 1)
        dx = Decimal.from_float(float(reference)) - times[j]
        slope = ((values[j + 1] - values[j]) / (times[j + 1] - times[j]) if reference >= self.times[0] and j + 1 < len(times) else Decimal(0))
        return integrals[j] + dx * (values[j] + slope * dx / 2)

    @property
    def phase_integrals(self):
        """Fractional integrated cycles at table knots, relative to the origin."""
        self._prepare_phase_data()
        return self._phase_integrals

    def phase_anchor(self, reference):
        """Host phase, frequency and interval for a local particle-time expansion.

        Only the scalar epoch uses extra precision. Particle arithmetic stays
        float64 on both backends; repeated reference/particle calls reuse it.
        """
        reference = float(reference)
        if self._phase_anchor is None or self._phase_anchor[0] != reference:
            self._prepare_phase_data()
            with localcontext() as ctx:
                ctx.prec = 80
                cycles = float((self._decimal_primitive(reference) - self._phase_origin) % 1) % 1.
            index = max(0, int(np.searchsorted(self.times, reference, side='right')) - 1)
            self._phase_anchor = reference, (cycles, float(self.value(reference)), index)
        return self._phase_anchor[1]

    def inverse_integral(self, cycles):
        """Invert a strictly positive frequency program, in physical seconds."""
        from scipy.optimize import brentq
        if np.any(self.values <= 0):
            raise ValueError("Clock frequency must be positive")
        span = abs(float(cycles)) / float(np.min(self.values)) + 1. / float(np.min(self.values))
        return brentq(lambda t: float(self.integral(t)) - cycles,
                      self.origin - span,
                      self.origin + span,
                      xtol=np.nextafter(0., 1.),
                      rtol=4 * np.finfo(float).eps)
