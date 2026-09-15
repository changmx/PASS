"""Prescribed piecewise-linear time programs for CPU and GPU consumers."""

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
        if (self.times.ndim != 1 or self.values.ndim != 1 or not len(self.times) or not len(self.values)
                or not np.all(np.isfinite(self.times)) or not np.all(np.isfinite(self.values))
                or not np.isfinite(self.origin) or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Programs require finite values and strictly increasing times")
        if len(self.values) == 1:
            self.times = np.asarray([origin], dtype=float)
        elif len(self.values) != len(self.times):
            raise ValueError("Program values and times must have equal lengths")
        self.slopes = np.r_[np.diff(self.values)/np.diff(self.times), 0.] if len(self.times)>1 else np.zeros(1)
        self.integrals = np.r_[0., np.cumsum(np.diff(self.times)*(self.values[1:]+self.values[:-1])/2)]
        self._devices = {}
        self.integral_origin = float(self._primitive(origin, 0., np))
        for array in (self.times, self.values, self.slopes, self.integrals):
            array.setflags(write=False)

    def _arrays(self, xp):
        if xp not in self._devices:
            self._devices[xp] = tuple(xp.asarray(a) for a in (self.times, self.values, self.slopes, self.integrals))
        return self._devices[xp]

    def _parts(self, reference, offset, xp):
        if xp is np and np.isscalar(offset):
            index = (0 if len(self.times) == 1 else
                     max(0, int(np.searchsorted(self.times-reference, offset, side='right'))-1))
            dx = (reference-self.times[index])+offset
            slope = 0. if (reference-self.times[0])+offset < 0. else self.slopes[index]
            return self.values[index], slope, dx, self.integrals[index]
        times, values, slopes, integrals = self._arrays(xp)
        index = xp.clip(xp.searchsorted(times-reference, offset, side='right')-1, 0, len(self.times)-1)
        dx = (reference-times[index])+offset
        slope = xp.where((reference-self.times[0])+offset < 0, 0., slopes[index])
        return values[index], slope, dx, integrals[index]

    def value(self, reference, offset=0., xp=np):
        if xp is np and np.isscalar(offset) and len(self.values) == 1:
            return self.values[0]
        value, slope, dx, _ = self._parts(reference, offset, xp)
        return value+slope*dx

    def _primitive(self, reference, offset, xp):
        value, slope, dx, base = self._parts(reference, offset, xp)
        return base+dx*(value+.5*slope*dx)

    def integral(self, reference, offset=0., xp=np):
        return self._primitive(reference, offset, xp)-self.integral_origin

    def phase_cycles(self, reference, offset=0., xp=np):
        # Reduce the scalar epoch first; preserve tiny intra-bunch differences.
        if len(self.values) == 1:
            base = np.remainder(self.values[0]*(reference-self.origin), 1.)
            return xp.remainder(base+self.values[0]*offset, 1.)
        value, slope, dx, base = self._parts(reference, offset, xp)
        return xp.remainder(xp.remainder(base-self.integral_origin, 1.)+dx*(value+.5*slope*dx), 1.)

    def inverse_integral(self, cycles):
        """Invert a strictly positive frequency program, in physical seconds."""
        from scipy.optimize import brentq
        if np.any(self.values <= 0):
            raise ValueError("Clock frequency must be positive")
        span = abs(float(cycles))/float(np.min(self.values))+1./float(np.min(self.values))
        return brentq(lambda t: float(self.integral(t))-cycles,
                      self.origin-span, self.origin+span, xtol=np.nextafter(0.,1.), rtol=4*np.finfo(float).eps)
