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


def evaluate_gpu(model, tau, longitudinal=True, *, primitive=False):
    return response_gpu(model, longitudinal).evaluate(tau, primitive=primitive)


def averaged_gpu(model, tau, width, longitudinal=True, memory_time=None):
    return response_gpu(model, longitudinal).evaluate(tau, width=width, memory_time=memory_time)


def device_arrays(owner, key, values):
    import cupy as cp
    cache = owner.__dict__.setdefault("_device_arrays", {})
    cache_key = (cp.cuda.runtime.getDevice(), key)
    if cache_key not in cache:
        cache[cache_key] = tuple(cp.asarray(v) for v in values)
    return cache[cache_key]


def response_gpu(model, longitudinal=True):
    """Cache compiled scalar response code; arrays and CUDA resources stay on device."""
    import cupy as cp
    cache = model.__dict__.setdefault('_raw_responses', {})
    key = (cp.cuda.runtime.getDevice(), bool(longitudinal))
    if key not in cache:
        cache[key] = DeviceResponse(model, longitudinal)
    return cache[key]


class DeviceResponse:
    """One scalar CUDA formula shared by evaluation, quadrature and direct sums.

    No pair matrix is materialized by direct tracking. CuPy supplies allocation,
    compilation and launch; all response arithmetic is inside the CUDA kernels.
    """
    def __init__(self, model, longitudinal):
        import cupy as cp
        body, data = _response_code(model, longitudinal)
        self.data = cp.asarray(data, dtype=cp.float64)
        self.code = ('#define M_PI 3.141592653589793238462643383279502884\n'
                     '#define INFINITY __longlong_as_double(0x7ff0000000000000LL)\n') + body + _RESPONSE_KERNELS
        self.kernels = {}

    def kernel(self, name, extra=''):
        import cupy as cp
        key = (name, extra)
        if key not in self.kernels:
            self.kernels[key] = cp.RawKernel(self.code+extra, name, options=('--std=c++17',))
        return self.kernels[key]

    def evaluate(self, tau, *, width=None, primitive=False, memory_time=None, scale=1.):
        import cupy as cp
        t = cp.asarray(tau, dtype=cp.float64)
        if width is None:
            out=cp.empty(t.shape,dtype=cp.float64)
            if out.size:self.kernel('response_flat')(((out.size+255)//256,),(256,),
                (cp.ascontiguousarray(t),self.data,np.int64(out.size),np.int32(primitive),
                 np.float64(np.inf if memory_time is None else memory_time),np.float64(scale),out))
            return out
        w = cp.asarray(0. if width is None else width, dtype=cp.float64)
        shape = np.broadcast_shapes(t.shape, w.shape)
        # broadcast_to is a view; only irregular public inputs need packing.
        t, w = cp.broadcast_to(t, shape), cp.broadcast_to(w, shape)
        key=(shape,tuple(s//8 for s in t.strides),tuple(s//8 for s in w.strides))
        layouts=self.__dict__.setdefault('layouts',{})
        if key not in layouts:
            if len(layouts)>=32:layouts.clear()
            layouts[key]=cp.asarray(key,dtype=cp.int64)
        tbase, wbase = t, w
        out = cp.empty(shape, dtype=cp.float64)
        if out.size:
            self.kernel('response_array')(((out.size+255)//256,), (256,),
                (tbase, wbase, self.data, layouts[key], np.int32(len(shape)), np.int64(out.size),
                 np.int32(1 if primitive else 2 if width is not None else 0),
                 np.float64(np.inf if memory_time is None else memory_time), np.float64(scale), out))
        return out


def _response_code(model, longitudinal):
    """Return device scalar formulas beside their CPU implementation."""
    from .wake_spectrum import RationalWakeModel, SpectrumWakeModel, spectrum_response_code
    prefix = '#include <cupy/complex.cuh>\nusing C=complex<double>;\n'
    if isinstance(model, (RationalWakeModel, SpectrumWakeModel)):
        return spectrum_response_code(model, longitudinal)
    values = []
    if isinstance(model, ConstantWakeModel):
        values = [model.amplitude, model.duration]
        formula = '''
        if(primitive)return d[0]*fmin(fmax(t,0.),d[1]);
        if(t<0.||t>d[1])return 0.;
        return d[0]*((t==0.||t==d[1])?.5:1.);
        '''
    elif isinstance(model, ResonatorWakeModel):
        values = [model.alpha, model.omega, model.r, model.q]
        formula = '''
        if(t<0.)return 0.;
        double a=d[0],w=d[1],r=d[2],q=d[3],c,s;
        if(q>.5){double v=w*sqrt((1.-.5/q)*(1.+.5/q)),x=v*t,e=exp(-a*t);
            c=e*cos(x);s=e*(x==0.?t:sin(x)/v);}
        else if(q==.5){c=exp(-a*t);s=c*t;}
        else {double v=a*sqrt((1.-2*q)*(1.+2*q)),e=exp(-w*w/(a+v)*t);
            c=e*(1.+exp(-2*v*t))*.5;s=-e*expm1(-2*v*t)/(2*v);}
        double value;
        if(primitive){
            if(LONGITUDINAL)value=2*a*r*s;
            else {double u=1.-c-a*s;
                if(fmax(a,w)*t<1.e-4)u=w*w*t*t*(.5-a*t/3.+(4*a*a-w*w)*t*t/24.+(4*a*w*w-8*a*a*a)*t*t*t/120.);
                value=2*a*r/w*u;}
        }else value=2*a*r*(LONGITUDINAL?c-a*s:w*s)*(t==0.?.5:1.);
        return value;
        '''
    elif isinstance(model, TabulatedWakeModel):
        n = len(model.times)
        values = np.r_[model.times, model.values, model.slopes, model.integrals]
        formula = f'''
        const int n={n};const double *x=d,*y=d+n,*s=d+2*n,*p=d+3*n-1;
        if(!primitive&&(t<x[0]||t>x[n-1]))return 0.;
        double u=fmin(fmax(t,x[0]),x[n-1]);int lo=0,hi=n-1;
        while(lo+1<hi){{int m=(lo+hi)/2;if(x[m]<=u)lo=m;else hi=m;}}
        double dx=u-x[lo];
        return primitive?p[lo]+y[lo]*dx+.5*s[lo]*dx*dx:
            (y[lo]+s[lo]*dx)*({int(model.causal)}&&t==0.?.5:1.);
        '''
    elif isinstance(model, ResistiveWallWakeModel):
        x, weight, (vi, wi), (vj, wj) = _wall_quadrature()
        nodes, gauss = np.polynomial.legendre.leggauss(24)
        v2, w2 = roots_genlaguerre(96, 1.5)
        values = np.r_[model.amplitude, model.time_scale, 2*const.c/model.radius**2,
                        x, weight, vi, wi, vj, wj, nodes, gauss, v2, w2]
        prefix += _WALL_DEVICE
        formula = '''
        if(t<0.)return 0.;double r=t/d[1],value;
        if(primitive)value=LONGITUDINAL?d[0]*d[1]*wall_first(r,d):d[2]*d[0]*d[1]*d[1]*wall_second(r,d);
        else {value=LONGITUDINAL?d[0]*(exp(-r)*cos(sqrt(3.)*r)/3.-sqrt(2.)/M_PI*wall_i(r,d)):
            d[2]*d[0]*d[1]*wall_first(r,d);if(t==0.)value*=.5;}
        return value;
        '''
    else:
        raise TypeError(f'No CUDA response implementation for {type(model).__name__}')
    if isinstance(model,(ConstantWakeModel,ResonatorWakeModel)):
        # Immutable model scalars specialize branches and oscillator constants
        # once at compilation, rather than recomputing them for each pair.
        for i,value in enumerate(values):formula=formula.replace(f'd[{i}]',float(value).hex())
    prefix += f'\n#define LONGITUDINAL {int(longitudinal)}\n'
    return prefix+'\n__device__ double response(double t,const double* d,bool primitive){'+formula+'}\n'+_AVERAGE_DEVICE, values


_AVERAGE_DEVICE = r'''
__device__ double averaged(double t,double w,const double* d,double horizon){
    if(w<=0.)return t<=horizon?response(t,d,false):0.;
    return (response(fmin(t+w*.5,horizon),d,true)-response(fmin(t-w*.5,horizon),d,true))/w;
}
'''


_WALL_DEVICE = r'''
__device__ double wall_i(double r,const double* d){
    double s=0.;if(r<1.){for(int j=0;j<160;j++){double x=d[3+j];s+=exp(-r*x*x)*d[163+j]*x*x;}return s;}
    for(int j=0;j<80;j++){double v=d[323+j]/r;s+=d[403+j]/(v*v*v+8.);}return s/(2*r*sqrt(r));
}
__device__ double wall_first(double r,const double* d){
    double s=0.,q=sqrt(3.)*r;
    if(r<1.e-4){for(int j=0;j<160;j++){double x=d[3+j];s-=expm1(-r*x*x)*d[163+j];}
        return (-expm1(-r)+exp(-r)*(2*pow(sin(q*.5),2)+sqrt(3.)*sin(q)))/12.-sqrt(2.)/M_PI*s;}
    if(r<1.){for(int j=0;j<160;j++){double x=d[3+j];s+=exp(-r*x*x)*d[163+j];}}
    else {for(int j=0;j<80;j++){double v=d[483+j]/r;s+=d[563+j]/(v*v*v+8.);}s/=2*sqrt(r);}
    return exp(-r)*(-cos(q)+sqrt(3.)*sin(q))/12.+sqrt(2.)/M_PI*s;
}
__device__ double wall_second(double r,const double* d){
    double s=0.;if(r<.1){for(int j=0;j<24;j++)s+=d[667+j]*wall_first((d[643+j]+1)*r*.5,d);return s*r*.5;}
    if(r<1.){for(int j=0;j<160;j++){double x=d[3+j];s-=expm1(-r*x*x)*d[163+j]/(x*x);}}
    else {for(int j=0;j<96;j++){double v=d[691+j]/r;s+=d[787+j]/(v*v*v+8.);}
        s=sqrt(M_PI*r)/8.-M_PI/(24*sqrt(2.))+s/(16*r*r*sqrt(r));}
    return (1.-exp(-r)*(cos(sqrt(3.)*r)+sqrt(3.)*sin(sqrt(3.)*r)))/24.+sqrt(2.)/M_PI*s;
}
'''


_RESPONSE_KERNELS = r'''
extern "C" __global__ void response_flat(const double* t,const double* data,long long n,
    int primitive,double horizon,double scale,double* out){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=scale*(!primitive&&t[i]>horizon?0.:response(t[i],data,primitive));
}
__device__ double coupling(double beta,const double* v,int n,int side){
    if(n==0)return 1.;if(n==1)return v[side];
    int lo=0,hi=n-1;
    while(lo+1<hi){int m=(lo+hi)/2;if(v[m]<=beta)lo=m;else hi=m;}
    double f=fmin(1.,fmax(0.,(beta-v[lo])/(v[lo+1]-v[lo])));
    return v[(side+1)*n+lo]*(1.-f)+v[(side+1)*n+lo+1]*f;
}
extern "C" __global__ void response_array(const double* t,const double* w,const double* data,
    const long long* layout,int ndim,long long n,int mode,double horizon,double scale,double* out){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    long long rem=i,ti=0,wi=0;
    for(int k=ndim-1;k>=0;k--){long long j=rem%layout[k];rem/=layout[k];ti+=j*layout[ndim+k];wi+=j*layout[2*ndim+k];}
    out[i]=scale*(mode==2?averaged(t[ti],w[wi],data,horizon):(mode==0&&t[ti]>horizon?0.:response(t[ti],data,mode==1)));
}
extern "C" __global__ void direct_response(const double* targets,const double* source,const double* widths,
    const double* moments,const double* beta,const double* target_beta,const double* velocity,int nv,
    int witness,const double* data,int nt,int ns,double scale,double horizon,int point,double* out){
    int lane=threadIdx.x&31,i=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;double sum=0.;
    if(i<nt)for(int j=lane;j<ns;j+=32){double tau=targets[i]-source[j];
        double r=point?(tau<=horizon?response(tau,data,false):0.):averaged(tau,widths[j],data,horizon);
        sum+=r*moments[j]*coupling(beta[j],velocity,nv,0);}
    for(int k=16;k;k/=2)sum+=__shfl_down_sync(0xffffffff,sum,k);
    if(lane==0&&i<nt)out[i]+=sum*scale*(witness?coupling(target_beta[i],velocity,nv,1):1.);
}
extern "C" __global__ void response_grid(const double* data,double step,double width,int offset,int n,
    double horizon,double scale,int point,double* out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    double t=(i-offset)*step;
    out[i]=scale*(point?(t<=horizon?response(t,data,false):0.):averaged(t,width,data,horizon));
}
'''
