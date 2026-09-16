"""Free-space parabolic slice fields, the spatial projection of a 4D waterbag.

The integrated density is 2 Q/(pi a b) * max(1-x^2/a^2-y^2/b^2, 0).
Its principal RMS widths are a/sqrt(6), b/sqrt(6). Fields have units V.
CPU and CUDA evaluate the same closed-form integral, including the exterior.
"""
from __future__ import annotations

import numpy as np

from PASS.utils.constants import const


def parabolic_elliptic_field(x, y, slice_charge, a, b, *, epsilon_0=const.epsilon0):
    """Return longitudinally integrated Ex, Ey for a parabolic elliptical slice.

    For A=sqrt(a^2+lambda), B=sqrt(b^2+lambda), lambda is zero inside
    and the positive confocal-ellipse root outside. The needed integrals are
    I_x=2/[A(A+B)], I_xx=2(2A+B)/[3A^3(A+B)^2],
    I_xy=2/[AB(A+B)^2], with Ex=Q*x/(pi*epsilon_0)*(I_x-x^2 I_xx-y^2 I_xy).
    Factoring I_x avoids differences of nearly equal semi-axes, so the circular
    limit needs no special tolerance or numerical quadrature.
    """
    try:
        a, b, charge, epsilon = map(float, (a, b, slice_charge, epsilon_0))
    except (TypeError, ValueError) as exc:
        raise ValueError("semi-axes, slice_charge and epsilon_0 must be finite scalars") from exc
    if not np.all(np.isfinite([a, b, charge, epsilon])) or min(a, b, epsilon) <= 0:
        raise ValueError("semi-axes and epsilon_0 must be positive finite; slice_charge must be finite")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("coordinates must be finite")
    aa, bb = a * a, b * b
    outside = (x / a)**2 + (y / b)**2 > 1
    lam = np.zeros_like(x)
    xx, yy = x[outside]**2, y[outside]**2
    linear = aa + bb - xx - yy
    constant = aa * bb - bb * xx - aa * yy
    root = np.sqrt(np.maximum(0., linear * linear - 4 * constant))
    # Rationalize the positive-linear branch close to the density edge.
    positive = linear >= 0
    value = np.empty_like(linear)
    value[positive] = -2 * constant[positive] / (linear[positive] + root[positive])
    value[~positive] = (root[~positive] - linear[~positive]) / 2
    lam[outside] = np.maximum(0., value)
    A, B = np.sqrt(aa + lam), np.sqrt(bb + lam)
    total = A + B
    rx2, ry2 = (x / A)**2, (y / B)**2
    factor = 2 * charge / (const.pi * epsilon * total)
    ex = factor * (x / A) * (1 - rx2 * (2*A+B)/(3*total) - ry2 * B/total)
    ey = factor * (y / B) * (1 - ry2 * (2*B+A)/(3*total) - rx2 * A/total)
    return ex, ey


def parabolic_round_field(x, y, slice_charge, radius, *, epsilon_0=const.epsilon0):
    """Circular special case; radius is the support edge, sqrt(6) times RMS."""
    return parabolic_elliptic_field(x, y, slice_charge, radius, radius, epsilon_0=epsilon_0)


PARABOLIC_CUDA = r"""
__device__ void parabolic_field(double x,double y,double q,double a,double b,double* ex,double* ey) {
    const double pi=3.14159265358979323846,eps=8.8541878128e-12;
    double aa=a*a,bb=b*b,lambda=0;
    if((x/a)*(x/a)+(y/b)*(y/b)>1) {
        double linear=aa+bb-x*x-y*y,constant=aa*bb-bb*x*x-aa*y*y;
        double root=sqrt(fmax(0.,linear*linear-4*constant));
        lambda=fmax(0.,linear>=0 ? -2*constant/(linear+root) : (root-linear)/2);
    }
    double A=sqrt(aa+lambda),B=sqrt(bb+lambda),sum=A+B;
    double rx=x/A,ry=y/B,f=2*q/(pi*eps*sum);
    *ex=f*rx*(1-rx*rx*(2*A+B)/(3*sum)-ry*ry*B/sum);
    *ey=f*ry*(1-ry*ry*(2*B+A)/(3*sum)-rx*rx*A/sum);
}
"""
