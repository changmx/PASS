"""SI impedance spectra and real rational responses.

F(f) = integral W(t) exp(-2 pi i f t) dt. Z_long = F; Z_transverse = i F.
Original samples are immutable. A finite-band inverse is two-sided, including
its truncation ringing. A causal projection is an explicit approximation.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import sici

from .wake_models import WakeModel
from .wake_conventions import require_cpu


def _readonly(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("Spectrum and mode arrays must be finite and one-dimensional")
    array.flags.writeable = False
    return array


def _sinc_derivative_factor(x):
    """(sin(x)-x*cos(x))/x**2, including the removable singularity."""
    small = np.abs(x) < 0.01
    safe = np.where(small, 1., x)
    return np.where(small, x*(1/3-x*x/30+x**4/840-x**6/45360),
                    (np.sin(x)-x*np.cos(x))/(safe*safe))


def _cin(x):
    """Integral from 0 to x of (cos(u)-1)/u; even and finite at zero."""
    x = np.abs(x)
    safe = np.where(x < 0.01, 1., x)
    return np.where(x < 0.01, -x*x/4+x**4/96-x**6/4320+x**8/322560,
                    sici(safe)[1]-np.euler_gamma-np.log(safe))


@dataclass(frozen=True)
class ImpedanceSpectrum:
    """Positive-frequency samples, linear interpolation, zero outside the band.

    A nonzero first frequency explicitly omits the unmeasured low-frequency
    band. No inferred DC value, padding, fitting or extrapolation is performed.
    Impedances are integrated over the physical element, in ohm/m**order.
    """
    frequencies: object
    real: object
    imag: object
    longitudinal: bool = True

    def __post_init__(self):
        for name in ("frequencies", "real", "imag"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        f = self.frequencies
        if (len(f) < 2 or f[0] < 0 or np.any(np.diff(f) <= 0)
                or self.real.shape != f.shape or self.imag.shape != f.shape):
            raise ValueError("Impedance needs matching samples at >=2 increasing nonnegative frequencies")
        if f[0] == 0:
            forbidden = self.imag[0] if self.longitudinal else self.real[0]
            if forbidden != 0:
                raise ValueError("DC impedance violates the real-wake Fourier symmetry")

    @property
    def values(self):
        return self.real+1j*self.imag

    @property
    def transfer(self):
        return self.values if self.longitudinal else -1j*self.values

    def impedance(self, frequencies):
        f = np.asarray(frequencies, float)
        positive = (np.interp(np.abs(f), self.frequencies, self.real, left=0, right=0)
                    + 1j*np.interp(np.abs(f), self.frequencies, self.imag, left=0, right=0))
        negative = positive.conj() if self.longitudinal else -positive.conj()
        return np.where(f < 0, negative, positive)

    def inverse(self, times):
        """Exact oscillatory integral of each linear frequency segment (Filon).

        Unlike a frequency-grid FFT this has no artificial periodic time window.
        Late-time accuracy still depends on the supplied frequency resolution.
        """
        t = np.asarray(times, float)
        output = np.empty(t.size)
        df = np.diff(self.frequencies)
        mid = (self.frequencies[1:]+self.frequencies[:-1])/2
        z = self.transfer
        mean, difference = (z[1:]+z[:-1])/2, (z[1:]-z[:-1])/2
        block = max(1, 131072//len(df))
        for begin in range(0, t.size, block):
            tau = t.ravel()[begin:begin+block, None]
            x = np.pi*tau*df
            integral = df*np.exp(2j*np.pi*tau*mid)*(
                mean*np.sinc(tau*df)+1j*difference*_sinc_derivative_factor(x))
            output[begin:begin+block] = 2*np.real(np.sum(integral, axis=1))
        return output.reshape(t.shape)

    def primitive(self, times):
        """Integral of the reconstructed wake from zero to t (signed t)."""
        t = np.asarray(times, float)
        output = np.empty(t.size)
        a, b = self.frequencies[:-1], self.frequencies[1:]
        df, mid = b-a, (b+a)/2
        z = self.transfer
        slope = np.diff(z)/df
        intercept = z[:-1]-slope*a
        nodes, weights = np.polynomial.legendre.leggauss(16)
        block = max(1, 65536//len(a))
        for begin in range(0, t.size, block):
            tau = t.ravel()[begin:begin+block, None]
            k = 2*np.pi*tau
            exponential = df*(np.exp(1j*k*mid)*np.sinc(tau*df)-1)
            logarithmic = (_cin(k*b)-_cin(k*a)
                           + 1j*(sici(k*b)[0]-sici(k*a)[0]))
            integral = (slope*exponential+intercept*logarithmic)/(2j*np.pi)
            # The analytic expression cancels near t=0 and for narrow high-f
            # intervals. Here the frequency integrand is smooth: fixed Gaussian
            # quadrature avoids that cancellation without resolving carrier turns.
            smooth = np.abs(tau*df) < 0.5
            rows, cols = np.nonzero(smooth)
            for start in range(0, len(rows), 8192):
                rr, cc = rows[start:start+8192], cols[start:start+8192]
                f = mid[cc, None]+df[cc, None]*nodes/2
                tt = tau[rr]
                zz = z[:-1][cc, None]+slope[cc, None]*(f-a[cc, None])
                # (exp(i*w*t)-1)/(i*w) = t*exp(i*w*t/2)*sinc(w*t/2).
                primitive = tt*np.exp(1j*np.pi*f*tt)*np.sinc(f*tt)
                integral[rr, cc] = df[cc]/2*np.sum(weights*zz*primitive, axis=1)
            output[begin:begin+block] = 2*np.real(np.sum(integral, axis=1))
        return output.reshape(t.shape)


class SpectrumWakeModel(WakeModel):
    def __init__(self, spectrum, reconstruction="two_sided"):
        if reconstruction not in {"two_sided", "causal_projection"}:
            raise ValueError("Choose two_sided or causal_projection spectrum reconstruction")
        self.spectrum = spectrum
        self.reconstruction = reconstruction
        self.causal = reconstruction == "causal_projection"

    def _check_plane(self, longitudinal):
        if longitudinal != self.spectrum.longitudinal:
            raise ValueError("Spectrum plane does not match its wake component")

    def evaluate(self, tau, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return evaluate_spectrum_gpu(self, tau, longitudinal)
        require_cpu(backend)
        self._check_plane(longitudinal)
        t = np.asarray(tau, float)
        value = self.spectrum.inverse(np.maximum(t, 0) if self.causal else t)
        if self.causal:
            value = np.where(t < 0, 0., np.where(t == 0, value/2, value))
        return value

    def primitive(self, tau, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return evaluate_spectrum_gpu(self, tau, longitudinal, primitive=True)
        require_cpu(backend)
        self._check_plane(longitudinal)
        return self.spectrum.primitive(np.maximum(tau, 0) if self.causal else tau)


class RationalWakeModel(WakeModel):
    """Real pole/residue response; explicit conjugates are stored.

    Left-half-plane poles propagate forward. Right-half-plane poles describe
    an anti-causal spatial branch and are only evaluated backwards; they must
    never be put in a temporal history recursion. No delta/derivative feedthrough
    is inferred. Zero-delay value is the mean of both one-sided limits.
    """
    def __init__(self, poles, residues, longitudinal=True, original_spectrum=None,
                 fit_diagnostics=None):
        self.poles = _readonly(poles, complex)
        self.residues = _readonly(residues, complex)
        self.longitudinal = longitudinal
        self.original_spectrum = original_spectrum
        self.fit_diagnostics = fit_diagnostics
        if (not len(self.poles) or self.residues.shape != self.poles.shape
                or np.any(self.poles.real == 0)):
            raise ValueError("Modes require matching poles/residues with nonzero real pole parts")
        remaining = list(range(len(self.poles)))
        while remaining:
            i = remaining.pop()
            if self.poles[i].imag == 0:
                if self.residues[i].imag != 0:
                    raise ValueError("Real poles require real residues")
            else:
                pairs = [j for j in remaining if self.poles[j] == self.poles[i].conjugate()
                         and self.residues[j] == self.residues[i].conjugate()]
                if not pairs:
                    raise ValueError("Complex poles and residues require explicit conjugate pairs")
                remaining.remove(pairs[0])
        self.causal = bool(np.all(self.poles.real < 0))

    def impedance(self, frequencies):
        s = 2j*np.pi*np.asarray(frequencies, float)
        response = np.sum(self.residues/(s[..., None]-self.poles), axis=-1)
        return response if self.longitudinal else 1j*response

    def _value(self, times, primitive=False):
        t = np.asarray(times, float)
        out = np.zeros(t.shape)
        for pole, residue in zip(self.poles, self.residues):
            forward = pole.real < 0
            local = np.maximum(t, 0) if forward else np.minimum(t, 0)
            value = (residue*np.expm1(pole*local)/pole if primitive
                     else residue*np.exp(pole*local))
            if primitive:
                out += (value.real if forward else -value.real)
            else:
                weight = np.where(t == 0, .5, (t > 0 if forward else t < 0).astype(float))
                out += (1 if forward else -1)*weight*value.real
        return out

    def evaluate(self, tau, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return evaluate_rational_gpu(self, tau, longitudinal)
        require_cpu(backend)
        if longitudinal != self.longitudinal:
            raise ValueError("Rational response plane does not match its component")
        return self._value(tau)

    def primitive(self, tau, longitudinal=True, backend="cpu"):
        if backend == "gpu":
            return evaluate_rational_gpu(self, tau, longitudinal, primitive=True)
        require_cpu(backend)
        if longitudinal != self.longitudinal:
            raise ValueError("Rational response plane does not match its component")
        return self._value(tau, primitive=True)

    def averaged(self, tau, width, longitudinal=True, backend="cpu"):
        """Integrate exponential segments without subtracting old primitives.

        The bin width can be many orders smaller than its delay in multi-turn
        tracking. expm1(p*width) retains that small interval exactly.
        """
        if backend == "gpu":
            from .wake_models import averaged_gpu
            return averaged_gpu(self, tau, width, longitudinal)
        else:
            require_cpu(backend)
            xp = np
        if longitudinal != self.longitudinal:
            raise ValueError("Rational response plane does not match its component")
        t, w = xp.broadcast_arrays(xp.asarray(tau, dtype=float), xp.asarray(width, dtype=float))
        if backend == "cpu" and np.any(w < 0):
            raise ValueError("Source bin width cannot be negative")
        out = xp.zeros(t.shape, dtype=float)
        for p, r in zip(self.poles, self.residues):
            forward = p.real < 0
            local = t if forward else -t
            pole = p if forward else -p
            begin = xp.maximum(local-w/2, 0.)
            span = xp.clip(local+w/2, 0., w)
            term = r*xp.exp(pole*begin)*xp.expm1(pole*span)/pole/xp.where(w > 0, w, 1.)
            out += (1 if forward else -1)*term.real
        return xp.where(w > 0, out, self.evaluate(t, longitudinal, backend))


def fit_spectrum(spectrum, initial_poles, *, optimize_poles=True, max_evaluations=500,
                 relative_floor=1e-3, tolerance=0.02):
    """Variable-projection rational fit with user-specified mode count and poles.

    Supply real poles and ONE pole with positive imaginary part per pair. Pole
    half-planes are fixed by the initial guesses; stable temporal poles cannot
    cross into a growing branch. Residues are fitted by scaled real least squares.
    This is not automatic model-order selection or a passivity guarantee.
    """
    poles = _readonly(initial_poles, complex)
    if (not len(poles) or np.any(poles.real == 0) or np.any(poles.imag < 0)
            or not 0 < relative_floor <= 1 or not tolerance > 0
            or max_evaluations < 1):
        raise ValueError("Invalid rational fit controls or initial poles")
    f = spectrum.frequencies
    scale_f = max(float(f[-1]*2*np.pi), 1.)
    target = spectrum.transfer
    scale_z = np.max(np.abs(target))
    if scale_z == 0:
        raise ValueError("A zero spectrum does not identify any poles")
    target = target/scale_z
    weight = 1/np.maximum(np.abs(target), relative_floor)
    s = 2j*np.pi*f/scale_f
    complex_mask = poles.imag > 0
    signs = np.sign(poles.real)
    x0 = np.r_[np.log(np.abs(poles.real)/scale_f),
                np.log(poles.imag[complex_mask]/scale_f)]

    def unpack(x):
        p = signs*np.exp(x[:len(poles)])+0j
        p[complex_mask] += 1j*np.exp(x[len(poles):])
        return p

    def solve(x, full=False):
        p = unpack(x)
        columns = []
        for pole in p:
            if pole.imag == 0:
                columns.append(1/(s-pole))
            else:
                a, b = 1/(s-pole), 1/(s-pole.conjugate())
                columns.extend((a+b, 1j*(a-b)))
        basis = np.asarray(columns).T
        matrix = np.concatenate((basis.real*weight[:, None], basis.imag*weight[:, None]))
        rhs = np.r_[target.real*weight, target.imag*weight]
        norms = np.linalg.norm(matrix, axis=0)
        coefficients = np.linalg.lstsq(matrix/norms, rhs, rcond=None)[0]/norms
        residual = matrix@coefficients-rhs
        return (p, coefficients, basis@coefficients) if full else residual

    result = (least_squares(solve, x0, max_nfev=max_evaluations, ftol=1e-12,
                            xtol=1e-12, gtol=1e-12, bounds=(x0-20, x0+20))
              if optimize_poles else None)
    p, coefficients, fitted = solve(result.x if result is not None else x0, full=True)
    error = np.abs(fitted-target)
    diagnostics = {
        "method": "variable_projection", "evaluations": 1 if result is None else result.nfev,
        "converged": True if result is None else bool(result.success),
        "relative_rms": float(np.linalg.norm(error)/np.linalg.norm(target)),
        "relative_max": float(np.max(error/np.maximum(np.abs(target), relative_floor))),
        "relative_floor": relative_floor, "tolerance": tolerance,
        "frequency_min_hz": float(f[0]), "frequency_max_hz": float(f[-1]),
        "passivity_enforced": False,
    }
    if not diagnostics["converged"] or diagnostics["relative_max"] > tolerance:
        raise ValueError(f"Rational fit failed the requested tolerance: {diagnostics}")
    expanded_poles, residues, ci = [], [], 0
    for pole in p:
        if pole.imag == 0:
            expanded_poles.append(pole*scale_f)
            residues.append(coefficients[ci]*scale_z*scale_f)
            ci += 1
        else:
            residue = (coefficients[ci]+1j*coefficients[ci+1])*scale_z*scale_f
            expanded_poles.extend((pole*scale_f, pole.conjugate()*scale_f))
            residues.extend((residue, residue.conjugate()))
            ci += 2
    return RationalWakeModel(expanded_poles, residues, spectrum.longitudinal,
                             original_spectrum=spectrum, fit_diagnostics=diagnostics)


# ----------------------------------------------------------------------------
# GPU: spectral inverse and rational responses
# ----------------------------------------------------------------------------


def spectrum_inverse_gpu(spectrum, times, primitive=False):
    from .wake_models import response_gpu
    model = spectrum.__dict__.get('_raw_inverse_model')
    if model is None:
        model = SpectrumWakeModel(spectrum)
        spectrum.__dict__['_raw_inverse_model'] = model
    return response_gpu(model, spectrum.longitudinal).evaluate(times, primitive=primitive)


def evaluate_rational_gpu(model, tau, longitudinal=True, *, primitive=False):
    from .wake_models import response_gpu
    return response_gpu(model, longitudinal).evaluate(tau, primitive=primitive)


def evaluate_spectrum_gpu(model, tau, longitudinal=True, *, primitive=False):
    from .wake_models import response_gpu
    return response_gpu(model, longitudinal).evaluate(tau, primitive=primitive)


def spectrum_response_code(model, longitudinal):
    """Exact segment integrals and stable pole sums, fused in scalar CUDA code."""
    prefix = r'''
    #include <cupy/complex.cuh>
    using C=complex<double>;
    __device__ C wake_expm1(C z){
        double h=sin(z.imag()*.5);
        return C(expm1(z.real())*cos(z.imag())-2*h*h,exp(z.real())*sin(z.imag()));
    }
    '''
    if isinstance(model, RationalWakeModel):
        if longitudinal != model.longitudinal:
            raise ValueError('Rational response plane does not match its component')
        values = np.column_stack((model.poles.real,model.poles.imag,model.residues.real,model.residues.imag)).ravel()
        body = f'#define NMODES {len(model.poles)}\n'+r'''
        __device__ double response(double t,const double* d,bool primitive){
            double out=0.;for(int j=0;j<NMODES;j++){
                C p(d[4*j],d[4*j+1]),r(d[4*j+2],d[4*j+3]);bool forward=p.real()<0.;
                double local=forward?fmax(t,0.):fmin(t,0.);
                C term=primitive?r*wake_expm1(p*local)/p:r*exp(p*local);
                double weight=primitive?1.:(t==0.?.5:(forward?t>0.:t<0.));
                out+=(forward?1.:-1.)*weight*term.real();
            }return out;
        }
        __device__ double averaged(double t,double w,const double* d,double horizon){
            if(w<=0.)return t<=horizon?response(t,d,false):0.;
            if(isfinite(horizon))return (response(fmin(t+w*.5,horizon),d,true)-response(fmin(t-w*.5,horizon),d,true))/w;
            double out=0.;for(int j=0;j<NMODES;j++){
                C p(d[4*j],d[4*j+1]),r(d[4*j+2],d[4*j+3]);bool forward=p.real()<0.;
                double local=forward?t:-t;C pole=forward?p:-p;
                double begin=fmax(local-w*.5,0.),span=fmin(w,fmax(local+w*.5,0.));
                out+=(forward?1.:-1.)*(r*exp(pole*begin)*wake_expm1(pole*span)/pole/w).real();
            }return out;
        }
        '''
    else:
        model._check_plane(longitudinal)
        s=model.spectrum;n=len(s.frequencies)
        nodes,weights=np.polynomial.legendre.leggauss(16)
        values=np.r_[s.frequencies,s.transfer.real,s.transfer.imag,nodes,weights]
        body=f'#define NPOINTS {n}\n#define CAUSAL {int(model.causal)}\n'+r'''
        #include <cupy/xsf/sici.h>
        __device__ double wake_sinc(double x){return x==0.?1.:sin(x)/x;}
        __device__ double wake_cin(double x){
            x=fabs(x);double q=x*x;
            if(x<.01)return q*(-.25+q*(1./96.+q*(-1./4320.+q/322560.)));
            double si,ci;xsf::sici(x,si,ci);return ci-0.5772156649015328606-log(x);
        }
        __device__ double response(double time,const double* d,bool primitive){
            if(CAUSAL&&time<0.)return 0.;double t=CAUSAL?fmax(time,0.):time,out=0.;
            const double *f=d,*zr=d+NPOINTS,*zi=d+2*NPOINTS,*nodes=d+3*NPOINTS,*weights=nodes+16;
            for(int j=0;j<NPOINTS-1;j++){
                double a=f[j],b=f[j+1],df=b-a,mid=(b+a)*.5;
                C z0(zr[j],zi[j]),z1(zr[j+1],zi[j+1]);
                C mean=(z0+z1)*.5,difference=(z1-z0)*.5,value;
                double x=M_PI*t*df;
                if(!primitive){
                    double q=x*x,factor=fabs(x)<.01?x*(1./3.-q/30.+q*q/840.-q*q*q/45360.):(sin(x)-x*cos(x))/q;
                    value=df*exp(C(0.,2*M_PI*t*mid))*(mean*wake_sinc(x)+C(0.,1.)*difference*factor);
                }else if(fabs(t*df)<.5){
                    value=C(0.,0.);C slope=(z1-z0)/df;
                    for(int k=0;k<16;k++){double ff=mid+df*.5*nodes[k],phase=M_PI*ff*t;
                        value+=weights[k]*(z0+slope*(ff-a))*t*exp(C(0.,phase))*wake_sinc(phase);}
                    value*=df*.5;
                }else{
                    double k=2*M_PI*t,sa,ca,sb,cb;xsf::sici(k*a,sa,ca);xsf::sici(k*b,sb,cb);
                    C slope=(z1-z0)/df,intercept=z0-slope*a;
                    C exponential=df*(exp(C(0.,k*mid))*wake_sinc(x)-C(1.,0.));
                    C logarithmic(wake_cin(k*b)-wake_cin(k*a),sb-sa);
                    value=(slope*exponential+intercept*logarithmic)/C(0.,2*M_PI);
                }out+=2*value.real();
            }return out*(CAUSAL&&!primitive&&time==0.?.5:1.);
        }
        '''
        from .wake_models import _AVERAGE_DEVICE
        body+=_AVERAGE_DEVICE
    return prefix+body,values
