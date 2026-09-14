"""Finite-beta round thick-wall impedance in the surface-impedance model.

G. Stupakov, PRAB 23, 094401 (2020), Eqs. (8), (19)-(20),
https://doi.org/10.1103/PhysRevAccelBeams.23.094401 .
Only the finite-conductivity correction is included: the identical perfect-wall
response has been subtracted. Neither direct space charge nor PEC image fields
are supplied by this class. The response assumes steady equal-velocity beams.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import i0e, i1e

from PASS.utils.constants import const
from .wake_spectrum import ImpedanceSpectrum


@dataclass(frozen=True)
class RoundWallImpedance:
    radius: float
    conductivity: float
    length: float
    skin_depth_ratio_max: float = .1
    surface_impedance_ratio_max: float = .1
    wall_thickness: float | None = None
    round_pipe = True

    def __post_init__(self):
        values = (self.radius, self.conductivity, self.length,
                  self.skin_depth_ratio_max, self.surface_impedance_ratio_max)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError("Round wall parameters and validity limits must be positive finite values")
        if self.skin_depth_ratio_max > .1 or self.surface_impedance_ratio_max > .1:
            raise ValueError("Surface-impedance validity limits must be <= 0.1")
        if self.wall_thickness is not None and not (np.isfinite(self.wall_thickness) and self.wall_thickness > 0):
            raise ValueError("Wall thickness must be positive and finite")

    def validity(self, frequencies, beta):
        f = np.asarray(frequencies, float)
        if not np.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError("Round-wall beta must lie in (0, 1]")
        if np.any(~np.isfinite(f)) or np.any(f == 0):
            raise ValueError("Surface-impedance wall needs nonzero finite frequencies; DC is outside its validity")
        omega = 2*np.pi*np.abs(f)
        depth = np.sqrt(2/(const.mu0*self.conductivity*omega))
        zeta = np.sqrt(omega*const.epsilon0/self.conductivity)
        size = self.radius if self.wall_thickness is None else min(self.radius, self.wall_thickness)
        ratio = depth/size
        if np.any(ratio > self.skin_depth_ratio_max):
            raise ValueError("Skin depth is too large for a thick-wall surface-impedance model; "
                             "use a finite-thickness field-matching impedance")
        if np.any(zeta > self.surface_impedance_ratio_max):
            raise ValueError("Surface impedance exceeds the good-conductor model validity")
        return {"max_skin_depth_ratio": float(np.max(ratio, initial=0)),
                "max_surface_impedance_ratio": float(np.max(zeta, initial=0)),
                "beta": float(beta), "content": "finite_conductivity_correction",
                "kinematics": "steady_equal_velocity"}

    def impedance(self, frequencies, beta, longitudinal=True):
        f = np.asarray(frequencies, float)
        self.validity(f, beta)
        w = 2*np.pi*np.abs(f)
        b = self.radius
        k = w/(beta*const.c)
        inverse_gamma2 = (1-beta)*(1+beta)
        x = b*k*np.sqrt(inverse_gamma2)
        scaled_i0 = i0e(x)
        # j = I1(x)/(x*I0(x)); scaled Bessel functions avoid overflow.
        small = x < 1e-3
        j = np.divide(i1e(x), x*scaled_i0, out=np.full_like(x, .5), where=x != 0)
        j = np.where(small, .5-x*x/16+x**4/96-11*x**6/6144, j)
        inverse_i0_squared = np.exp(-2*x)/scaled_i0**2
        zeta = (1-1j)*np.sqrt(w*const.epsilon0/(2*self.conductivity))
        if longitudinal:
            cgs = 2j*zeta/(const.c*b)*inverse_i0_squared/(1j+zeta*w*b/const.c*j)
        else:
            # Algebraic reduction of Eq. (20), including gamma -> infinity.
            # gamma**2*(2*j-1) otherwise loses every significant digit near x=0.
            gterm = np.where(small, (b*k)**2*(-1/8+x*x/48-11*x**4/3072),
                             (2*j-1)/max(inverse_gamma2, np.finfo(float).tiny))
            numerator = zeta*(b*zeta*k*j*inverse_gamma2+1j*beta*(j-1))
            denominator = const.c*j*b*b*(
                zeta*j*j*(1+x*x)+1j*b*k*beta*(zeta*zeta+1)*j*(j-1)
                + zeta*beta*beta*gterm)
            cgs = numerator/denominator*inverse_i0_squared
        # The paper uses exp(-i omega t) for the inverse. PASS uses the opposite
        # Fourier exponent for both longitudinal and transverse impedances.
        positive = np.conjugate(cgs)*(const.mu0*const.c**2/(4*np.pi))*self.length
        negative = positive.conjugate() if longitudinal else -positive.conjugate()
        return np.where(f < 0, negative, positive)

    def spectrum(self, frequencies, beta, longitudinal=True):
        f = np.asarray(frequencies, float)
        values = self.impedance(f, beta, longitudinal)
        return ImpedanceSpectrum(f, values.real, values.imag, longitudinal)
