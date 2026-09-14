"""Integrated causal time-domain wakes: seconds and V/C/m**(source+test order).

Longitudinal wake is positive energy loss; transverse wake is positive force.
The impedance convention for future conversion is Zz=integral(Wz exp(-iwt)dt)
and Zperp=i*integral(Wperp exp(-iwt)dt). No Fourier factor enters a time FFT.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import roots_genlaguerre

from PASS.utils.constants import const
from .wake_conventions import require_cpu


class WakeModel:
    causal = True

    def evaluate(self, tau, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return evaluate_gpu(self, tau, longitudinal)
        require_cpu(backend)
        tau = np.asarray(tau, dtype=float)
        value = self._positive(np.maximum(tau, 0), longitudinal)
        # Symmetric point-charge convention at a finite causal jump.
        return np.where(tau < 0, 0.0, np.where(tau == 0, value * 0.5, value))

    def primitive(self, tau, longitudinal=True, backend="cpu"):
        require_cpu(backend)
        raise NotImplementedError

    def averaged(self, tau, width, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return averaged_gpu(self, tau, width, longitudinal)
        require_cpu(backend)
        tau, width = np.broadcast_arrays(np.asarray(tau, float), np.asarray(width, float))
        if np.any(width < 0):
            raise ValueError("Source bin width cannot be negative")
        result = np.array(self.evaluate(tau, longitudinal), copy=True)
        finite = width > 0
        if np.any(finite):
            t, h = tau[finite], width[finite] * 0.5
            result[finite] = (self.primitive(t + h, longitudinal)
                              - self.primitive(t - h, longitudinal)) / (2 * h)
        return result

    def validate_beta(self, beta):
        if not 0 < beta <= 1:
            raise ValueError("Wake source beta must lie in (0, 1]")


@dataclass(frozen=True)
class ConstantWakeModel(WakeModel):
    amplitude: float
    duration: float

    def __post_init__(self):
        if not np.isfinite(self.amplitude) or not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Constant wake requires finite amplitude and positive finite duration")

    def _positive(self, t, longitudinal):
        return self.amplitude * np.where(t < self.duration, 1.0,
                                        np.where(t == self.duration, 0.5, 0.0))

    def primitive(self, tau, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return evaluate_gpu(self, tau, longitudinal, primitive=True)
        require_cpu(backend)
        return self.amplitude * np.clip(tau, 0, self.duration)


@dataclass(frozen=True)
class ResonatorWakeModel(WakeModel):
    r: float
    q: float
    frequency: float

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.r, self.q, self.frequency)):
            raise ValueError("Resonator r, q and frequency must be finite and positive")

    @property
    def omega(self):
        return 2 * np.pi * self.frequency

    @property
    def alpha(self):
        return self.omega / (2 * self.q)

    def _cs(self, t):
        w, a = self.omega, self.alpha
        if self.q > 0.5:
            d = w * np.sqrt((1 - 0.5 / self.q) * (1 + 0.5 / self.q))
            decay = np.exp(-a * t)
            return decay * np.cos(d*t), decay * t * np.sinc(d*t / np.pi)
        if self.q == 0.5:
            decay = np.exp(-a * t)
            return decay, decay * t
        d = a * np.sqrt((1 - 2*self.q) * (1 + 2*self.q))
        slow = -w*w / (a + d)
        e = np.exp(slow*t)
        # exp(slow*t)-exp(fast*t), stable at t=0 and for very small Q.
        return e * (1 + np.exp(-2*d*t)) / 2, -e * np.expm1(-2*d*t) / (2*d)

    def _positive(self, t, longitudinal):
        c, s = self._cs(t)
        return 2*self.alpha*self.r * (c-self.alpha*s if longitudinal else self.omega*s)

    def primitive(self, tau, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return evaluate_gpu(self, tau, longitudinal, primitive=True)
        require_cpu(backend)
        t = np.maximum(np.asarray(tau, float), 0)
        c, s = self._cs(t)
        if longitudinal:
            return 2*self.alpha*self.r*s
        value = 1-c-self.alpha*s
        small = np.maximum(self.alpha, self.omega)*t < 1e-4
        # Integral of oscillator impulse response, without subtracting 1-1.
        a, w = self.alpha, self.omega
        series = w*w*t*t*(0.5-a*t/3+(4*a*a-w*w)*t*t/24
                           +(4*a*w*w-8*a**3)*t**3/120)
        return 2*a*self.r/w * np.where(small, series, value)


class TabulatedWakeModel(WakeModel):
    def __init__(self, times, values, causal=True):
        self.causal = causal
        self.times = np.array(times, dtype=float, copy=True)
        self.values = np.array(values, dtype=float, copy=True)
        if (self.times.ndim != 1 or self.times.size < 2 or self.times.shape != self.values.shape
                or not np.all(np.isfinite(self.times)) or not np.all(np.isfinite(self.values))
                or (causal and self.times[0] != 0) or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Tabulated wake needs finite matching increasing times; causal data start at zero")
        self.slopes = np.diff(self.values) / np.diff(self.times)
        self.integrals = np.r_[0, np.cumsum(np.diff(self.times)*(self.values[:-1]+self.values[1:])/2)]
        for array in (self.times, self.values, self.slopes, self.integrals):
            array.flags.writeable = False

    def _positive(self, t, longitudinal):
        return np.interp(t, self.times, self.values, right=0.0)

    def evaluate(self, tau, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return evaluate_gpu(self, tau, longitudinal)
        if self.causal:
            return super().evaluate(tau, longitudinal, backend)
        require_cpu(backend)
        return np.interp(tau, self.times, self.values, left=0., right=0.)

    def primitive(self, tau, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return evaluate_gpu(self, tau, longitudinal, primitive=True)
        require_cpu(backend)
        t = np.clip(tau, self.times[0], self.times[-1])
        i = np.clip(np.searchsorted(self.times, t, side="right")-1, 0, len(self.slopes)-1)
        d = t-self.times[i]
        return self.integrals[i] + self.values[i]*d + self.slopes[i]*d*d/2


@lru_cache(maxsize=1)
def _wall_quadrature():
    nodes, weights = np.polynomial.legendre.leggauss(160)
    angle = (nodes+1)*np.pi/4
    x = np.tan(angle)
    weight = weights*np.pi/4/(np.cos(angle)**2)/(x**6+8)
    return x, weight, roots_genlaguerre(80, 0.5), roots_genlaguerre(80, -0.5)


@dataclass(frozen=True)
class ResistiveWallWakeModel(WakeModel):
    """Bane-Sands DC, round thick wall, ultrarelativistic integrated wake.

    s0=(2*b**2/(Z0*sigma))**(1/3). Includes the finite short-distance wake;
    does not regularize a long-distance power law with an arbitrary epsilon.
    """
    radius: float
    conductivity: float
    length: float
    round_pipe = True

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.radius, self.conductivity, self.length)):
            raise ValueError("Wall radius, conductivity and length must be finite and positive")

    @property
    def time_scale(self):
        return (2*self.radius**2/(const.mu0*const.c*self.conductivity))**(1/3)/const.c

    @property
    def amplitude(self):
        return 4*const.mu0*const.c**2*self.length/(np.pi*self.radius**2)

    def validate_beta(self, beta):
        super().validate_beta(beta)
        if beta < 0.99:
            raise ValueError("DC round-wall wake is ultrarelativistic (requires beta >= 0.99)")

    def _integrals(self, r):
        shape = np.shape(r)
        r = np.asarray(r).reshape(-1)
        x, weight, lag_i, lag_j = _wall_quadrature()
        i, j = np.empty_like(r), np.empty_like(r)
        small = r < 1
        # Bound temporary storage independently of particle/slice count.
        for ids in np.array_split(np.flatnonzero(small), max(1, int(small.sum())//2048+1)):
            exp = np.exp(-r[ids, None]*x*x)
            i[ids] = exp @ (weight*x*x)
            j[ids] = exp @ weight
        for ids in np.array_split(np.flatnonzero(~small), max(1, int((~small).sum())//2048+1)):
            rr = r[ids, None]
            v, weights = lag_i
            i[ids] = np.sum(weights/((v/rr)**3+8), axis=1)/(2*r[ids]**1.5)
            v, weights = lag_j
            j[ids] = np.sum(weights/((v/rr)**3+8), axis=1)/(2*np.sqrt(r[ids]))
        return i.reshape(shape), j.reshape(shape)

    def _long_primitive(self, t):
        r = np.asarray(t)/self.time_scale
        _, j = self._integrals(r)
        value = (np.exp(-r)*(-np.cos(np.sqrt(3)*r)+np.sqrt(3)*np.sin(np.sqrt(3)*r))/12
                 + np.sqrt(2)/np.pi*j)
        # Near the origin evaluate integral(1-exp(-r*x*x)) using expm1.
        small = r < 1e-4
        if np.any(small):
            x, weight, _, _ = _wall_quadrature()
            rr = r[small]
            osc = (-np.expm1(-rr) + np.exp(-rr)*(2*np.sin(np.sqrt(3)*rr/2)**2
                    + np.sqrt(3)*np.sin(np.sqrt(3)*rr)))/12
            integral = -np.expm1(-rr[:, None]*x*x) @ weight
            value = np.array(value, copy=True)
            value[small] = osc - np.sqrt(2)/np.pi*integral
        return self.amplitude*self.time_scale*value

    def _positive(self, t, longitudinal):
        if not longitudinal:
            return 2*const.c/self.radius**2*self._long_primitive(t)
        r = t/self.time_scale
        i, _ = self._integrals(r)
        return self.amplitude*(np.exp(-r)*np.cos(np.sqrt(3)*r)/3-np.sqrt(2)/np.pi*i)

    @lru_cache(maxsize=8192)
    def _second_integral(self, r):
        if r == 0:
            return 0.0
        if r < 0.1:
            # The closed form cancels two O(r) terms to obtain O(r**2).
            # Integrate the cancellation-safe first primitive near zero.
            nodes, weights = np.polynomial.legendre.leggauss(24)
            first = self._long_primitive((nodes+1)*r*self.time_scale/2)
            return float(weights @ first)*r/(2*self.amplitude*self.time_scale)
        # Subtract the boundary layer analytically. Direct infinite-interval
        # quadrature loses the constant tail for r >~ 1e7, even after scaling.
        if r >= 1:
            correction = quad(lambda u: u**4*np.exp(-u*u)/((u/np.sqrt(r))**6+8),
                              0, np.inf, epsabs=1e-12, epsrel=2e-12)[0]/r**2.5
            integral = np.sqrt(np.pi*r)/8-np.pi/(24*np.sqrt(2))+correction/8
            oscillatory = (1-np.exp(-r)*(np.cos(np.sqrt(3)*r)+np.sqrt(3)*np.sin(np.sqrt(3)*r)))/24
            return oscillatory + np.sqrt(2)/np.pi*integral
        # Scaling x=u/sqrt(r) resolves the transition region.
        def integrand(u):
            if u == 0:
                return np.sqrt(r)/8
            return np.sqrt(r)*(-np.expm1(-u*u))/(u*u*((u/np.sqrt(r))**6+8))
        integral = quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=2e-10)[0]
        oscillatory = (1-np.exp(-r)*(np.cos(np.sqrt(3)*r)+np.sqrt(3)*np.sin(np.sqrt(3)*r)))/24
        return oscillatory + np.sqrt(2)/np.pi*integral

    def primitive(self, tau, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return evaluate_gpu(self, tau, longitudinal, primitive=True)
        require_cpu(backend)
        t = np.maximum(np.asarray(tau, float), 0)
        if longitudinal:
            return self._long_primitive(t)
        r = t/self.time_scale
        out = np.array([self._second_integral(float(v)) for v in r.flat]).reshape(r.shape)
        return 2*const.c/self.radius**2*self.amplitude*self.time_scale**2*out


# ----------------------------------------------------------------------------
# GPU: device responses and bin integration
# ----------------------------------------------------------------------------


def _oscillator_gpu(model, t):
    import cupy as cp
    w, a = model.omega, model.alpha
    if model.q > .5:
        d = w*np.sqrt((1-.5/model.q)*(1+.5/model.q))
        decay = cp.exp(-a*t)
        return decay*cp.cos(d*t), decay*t*cp.sinc(d*t/np.pi)
    if model.q == .5:
        decay = cp.exp(-a*t)
        return decay, decay*t
    d = a*np.sqrt((1-2*model.q)*(1+2*model.q))
    slow = -w*w/(a+d)
    e = cp.exp(slow*t)
    return e*(1+cp.exp(-2*d*t))/2, -e*cp.expm1(-2*d*t)/(2*d)


def _wall_integrals_gpu(model, r):
    import cupy as cp
    x, weight, lag_i, lag_j = _wall_quadrature()
    x, weight, vi, wi, vj, wj = device_arrays(model, "wall_quadrature", (x, weight, *lag_i, *lag_j))
    shape = r.shape
    flat = r.ravel()
    first, second = cp.empty_like(flat), cp.empty_like(flat)
    for start in range(0, flat.size, 1024):
        rr = flat[start:start+1024, None]
        safe = cp.maximum(rr, 1.)
        small_i = cp.sum(cp.exp(-rr*x*x)*weight*x*x, axis=1)
        small_j = cp.sum(cp.exp(-rr*x*x)*weight, axis=1)
        large_i = cp.sum(wi/((vi/safe)**3+8), axis=1)/(2*safe[:, 0]**1.5)
        large_j = cp.sum(wj/((vj/safe)**3+8), axis=1)/(2*cp.sqrt(safe[:, 0]))
        first[start:start+1024] = cp.where(rr[:, 0] < 1, small_i, large_i)
        second[start:start+1024] = cp.where(rr[:, 0] < 1, small_j, large_j)
    return first.reshape(shape), second.reshape(shape)


def _wall_long_primitive_gpu(model, t):
    import cupy as cp
    r = t/model.time_scale
    _, j = _wall_integrals_gpu(model, r)
    value = (cp.exp(-r)*(-cp.cos(np.sqrt(3)*r)+np.sqrt(3)*cp.sin(np.sqrt(3)*r))/12
             + np.sqrt(2)/np.pi*j)
    x, weight, _, _ = _wall_quadrature()
    x, weight = device_arrays(model, "wall_small", (x, weight))
    flat, result = r.ravel(), value.ravel()
    for start in range(0, flat.size, 1024):
        rr = flat[start:start+1024]
        integral = cp.sum(-cp.expm1(-rr[:, None]*x*x)*weight, axis=1)
        osc = (-cp.expm1(-rr)+cp.exp(-rr)*(2*cp.sin(np.sqrt(3)*rr/2)**2+np.sqrt(3)*cp.sin(np.sqrt(3)*rr)))/12
        result[start:start+1024] = cp.where(rr < 1e-4, osc-np.sqrt(2)/np.pi*integral, result[start:start+1024])
    return model.amplitude*model.time_scale*result.reshape(r.shape)


def _wall_second_gpu(model, t):
    import cupy as cp
    from scipy.special import roots_genlaguerre
    r = t/model.time_scale
    x, weight, _, _ = _wall_quadrature()
    nodes, gauss = np.polynomial.legendre.leggauss(24)
    v, lag = roots_genlaguerre(96, 1.5)
    x, weight, nodes, gauss, v, lag = device_arrays(model, "wall_second", (x, weight, nodes, gauss, v, lag))
    flat, out = r.ravel(), cp.empty(r.size)
    for start in range(0, flat.size, 256):
        rr = flat[start:start+256, None]
        safe = cp.maximum(rr, 1.)
        integral_small = cp.sum(-cp.expm1(-rr*x*x)*weight/(x*x), axis=1)
        # Analytic subtraction of the r->infinity boundary layer. This avoids
        # needing quadrature nodes out to sqrt(r) in the scaled integral.
        correction = cp.sum(lag/((v/safe)**3+8), axis=1)/(2*safe[:, 0]**2.5)
        integral_large = cp.sqrt(np.pi*safe[:, 0])/8-np.pi/(24*np.sqrt(2))+correction/8
        integral = cp.where(rr[:, 0] < 1, integral_small, integral_large)
        osc = (1-cp.exp(-rr[:, 0])*(cp.cos(np.sqrt(3)*rr[:, 0])+np.sqrt(3)*cp.sin(np.sqrt(3)*rr[:, 0])))/24
        ordinary = osc+np.sqrt(2)/np.pi*integral
        small_r = cp.minimum(rr, .1)
        first = _wall_long_primitive_gpu(model, (nodes+1)*small_r*model.time_scale/2)
        small = cp.sum(gauss*first, axis=1)*small_r[:, 0]/(2*model.amplitude*model.time_scale)
        out[start:start+256] = cp.where(rr[:, 0] < .1, small, ordinary)
    from PASS.utils.constants import const
    return 2*const.c/model.radius**2*model.amplitude*model.time_scale**2*out.reshape(r.shape)


def evaluate_gpu(model, tau, longitudinal=True, *, primitive=False):
    import cupy as cp
    from .wake_spectrum import SpectrumWakeModel, RationalWakeModel, evaluate_spectrum_gpu, evaluate_rational_gpu
    t = cp.asarray(tau, dtype=cp.float64)
    positive = cp.maximum(t, 0.)
    if isinstance(model, SpectrumWakeModel):
        return evaluate_spectrum_gpu(model, t, longitudinal, primitive=primitive)
    elif isinstance(model, RationalWakeModel):
        return evaluate_rational_gpu(model, t, longitudinal, primitive=primitive)
    elif isinstance(model, ConstantWakeModel):
        value = (model.amplitude*cp.clip(t, 0, model.duration) if primitive else
                 model.amplitude*cp.where(positive < model.duration, 1., cp.where(positive == model.duration, .5, 0.)))
    elif isinstance(model, ResonatorWakeModel):
        c, s = _oscillator_gpu(model, positive)
        a, w = model.alpha, model.omega
        if primitive:
            if longitudinal:
                value = 2*a*model.r*s
            else:
                small = max(a, w)*positive < 1e-4
                series = w*w*positive**2*(.5-a*positive/3+(4*a*a-w*w)*positive**2/24
                        +(4*a*w*w-8*a**3)*positive**3/120)
                value = 2*a*model.r/w*cp.where(small, series, 1-c-a*s)
        else:
            value = 2*a*model.r*(c-a*s if longitudinal else w*s)
    elif isinstance(model, TabulatedWakeModel):
        times, values, slopes, integrals = device_arrays(model, "table", (model.times, model.values, model.slopes, model.integrals))
        if primitive:
            local = cp.clip(t, times[0], times[-1])
            i = cp.clip(cp.searchsorted(times, local, side="right")-1, 0, len(slopes)-1)
            d = local-times[i]
            return integrals[i]+values[i]*d+slopes[i]*d*d/2
        value = cp.interp(positive if model.causal else t, times, values, left=0., right=0.)
    elif isinstance(model, ResistiveWallWakeModel):
        if primitive:
            return _wall_long_primitive_gpu(model, positive) if longitudinal else _wall_second_gpu(model, positive)
        if longitudinal:
            r = positive/model.time_scale
            i, _ = _wall_integrals_gpu(model, r)
            value = model.amplitude*(cp.exp(-r)*cp.cos(np.sqrt(3)*r)/3-np.sqrt(2)/np.pi*i)
        else:
            from PASS.utils.constants import const
            value = 2*const.c/model.radius**2*_wall_long_primitive_gpu(model, positive)
    else:
        raise TypeError(f"No GPU response implementation for {type(model).__name__}")
    if primitive or not model.causal:
        return value
    return cp.where(t < 0, 0., cp.where(t == 0, value/2, value))


def averaged_gpu(model, tau, width, longitudinal=True, memory_time=None):
    import cupy as cp
    from .wake_spectrum import RationalWakeModel
    if isinstance(model, RationalWakeModel) and memory_time is None:
        return model.averaged(tau, width, longitudinal, backend="gpu")
    t, w = cp.broadcast_arrays(cp.asarray(tau, dtype=cp.float64), cp.asarray(width, dtype=cp.float64))
    point = evaluate_gpu(model, t, longitudinal)
    upper, lower = t+w/2, t-w/2
    if memory_time is not None:
        point = cp.where(t <= memory_time, point, 0.)
        upper, lower = cp.minimum(upper, memory_time), cp.minimum(lower, memory_time)
    finite = (evaluate_gpu(model, upper, longitudinal, primitive=True)-evaluate_gpu(model, lower, longitudinal, primitive=True))/cp.where(w > 0, w, 1.)
    return cp.where(w > 0, finite, point)


def device_arrays(owner, key, values):
    import cupy as cp
    cache = owner.__dict__.setdefault("_device_arrays", {})
    cache_key = (cp.cuda.runtime.getDevice(), key)
    if cache_key not in cache:
        cache[cache_key] = tuple(cp.asarray(v) for v in values)
    return cache[cache_key]
