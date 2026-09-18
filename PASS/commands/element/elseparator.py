"""Electrostatic septum: a field gap, infinite-height plates and vacuum walls.

A roll projects the geometry and the kick; it never changes the supplied L.
Positive length uses a relativistic uniform-field body map and hard-edge
potential matching. Zero length uses an effective kick with the incident
longitudinal speed. V is a voltage difference; VL is its integral in V m.
Particles stop at first contact with material, including curved trajectories.
"""
from dataclasses import dataclass
import logging
import math

import numpy as np

from PASS.commands.command import Command
from PASS.commands.element.drift import drift_factors, drift_cuda_factors
from PASS.para.schema.elements import ElSeparatorItem
from PASS.para.schema.space_charge import parse_element_space_charge
from PASS.utils.aperture import (
    build_aperture,
    AllSpaceAperture,
    RectangleAperture,
    EllipticAperture,
    IntersectionAperture,
    RacetrackAperture,
    OctagonAperture,
    PolygonAperture,
)
from PASS.utils.constants import const
from PASS.utils.slicing import configure_element_slicing, print_element_slicing, run_body_slices


def _aperture_primitives(geometry):
    """Return finite line segments and (cx, cy, a, b, side) ellipse arcs."""
    if isinstance(geometry, AllSpaceAperture):
        return [], []
    if isinstance(geometry, EllipticAperture):
        return [], [(0., 0., geometry.a, geometry.b, 0)]
    if isinstance(geometry, IntersectionAperture):
        parts = [_aperture_primitives(g) for g in geometry.apertures]
        return [s for p in parts for s in p[0]], [e for p in parts for e in p[1]]
    if isinstance(geometry, RacetrackAperture):
        w, h, a, b = geometry.w, geometry.h, geometry.a, geometry.b
        segments = [(-w, -h, w, -h), (-w, h, w, h)]
        if h != b:
            lo, hi = min(h, b), max(h, b)
            segments += [(x, sign * lo, x, sign * hi) for x in (-w, w) for sign in (-1, 1)]
        return segments, [(-w, 0., a, b, -1), (w, 0., a, b, 1)]
    if isinstance(geometry, RectangleAperture):
        x0, x1, y0, y1 = geometry.x_min, geometry.x_max, geometry.y_min, geometry.y_max
        vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    elif isinstance(geometry, OctagonAperture):
        w, h, d = geometry.w, geometry.h, geometry.d
        vertices = [(-w + d, -h), (w - d, -h), (w, -h + d), (w, h - d), (w - d, h), (-w + d, h), (-w, h - d), (-w, -h + d)]
    elif isinstance(geometry, PolygonAperture):
        vertices = list(geometry.vertices)
    else:
        raise TypeError(f"Unsupported aperture geometry: {type(geometry).__name__}")
    return [(*a, *b) for a, b in zip(vertices, vertices[1:] + vertices[:1]) if a != b], []


def _aperture_inside_cuda(geometry, name):
    """Generate the existing strict point predicate, using double geometry."""

    def f(x):
        return format(float(x), '.17g')

    prefix = f'''__device__ inline bool {name}(
    double x,
    double y
) {{
'''
    if isinstance(geometry, AllSpaceAperture):
        body = 'return true;'
    elif isinstance(geometry, RectangleAperture):
        body = (f'return x>{f(geometry.x_min)} && x<{f(geometry.x_max)} '
                f'&& y>{f(geometry.y_min)} && y<{f(geometry.y_max)};')
    elif isinstance(geometry, EllipticAperture):
        body = f'return (x/{f(geometry.a)})*(x/{f(geometry.a)})+(y/{f(geometry.b)})*(y/{f(geometry.b)})<1.;'
    elif isinstance(geometry, IntersectionAperture):
        definitions = [_aperture_inside_cuda(g, f'{name}_{i}') for i, g in enumerate(geometry.apertures)]
        body = 'return ' + ' && '.join(f'{name}_{i}(x,y)' for i in range(len(definitions))) + ';'
        return '\n'.join(definitions) + '\n' + prefix + body + ' }\n'
    elif isinstance(geometry, RacetrackAperture):
        w, h, a, b = map(f, (geometry.w, geometry.h, geometry.a, geometry.b))
        body = (f'double ax=fabs(x), dx=(ax-({w}))/({a}); '
                f'return (ax<({w}) && fabs(y)<({h})) || '
                f'(ax>({w}) && dx*dx+(y/({b}))*(y/({b}))<1.) || '
                f'(ax==({w}) && fabs(y)<fmin({h},{b}));')
    elif isinstance(geometry, OctagonAperture):
        body = (f'return fabs(x)<{f(geometry.w)} && fabs(y)<{f(geometry.h)} '
                f'&& fabs(x)+fabs(y)<{f(geometry.w+geometry.h-geometry.d)};')
    elif isinstance(geometry, PolygonAperture):
        vertices = list(geometry.vertices)
        tolerance = 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(vertices))))
        body = f'bool inside=false; const double tol={f(tolerance)};\n'
        for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1]):
            dx, dy = x2 - x1, y2 - y1
            if dx == dy == 0:
                continue
            body += (f'if (fabs(({f(dx)})*(y-({f(y1)}))-({f(dy)})*(x-({f(x1)})))'
                     f'/{f(np.hypot(dx,dy))}<=tol && x>={f(min(x1,x2))}-tol '
                     f'&& x<={f(max(x1,x2))}+tol && y>={f(min(y1,y2))}-tol '
                     f'&& y<={f(max(y1,y2))}+tol) return false;\n')
            if dy != 0:
                body += (f'if ((({f(y1)}>y)!=({f(y2)}>y)) && '
                         f'x<({f(dx)})*(y-({f(y1)}))/({f(dy)})+({f(x1)})) inside=!inside;\n')
        body += 'return inside;'
    else:
        raise TypeError(type(geometry).__name__)
    return prefix + body + ' }\n'


@dataclass
class _ApertureBoundary:
    """First intersections with the existing, extruded vacuum apertures.

    Distances are along a straight tracking segment, not Euclidean path lengths.
    CPU predicates and generated CUDA use the same boundary primitives.  A wall
    touch is a loss, including tangencies and paths through a non-convex polygon.
    """

    geometry: object

    def __post_init__(self):
        self.segments, self.ellipses = _aperture_primitives(self.geometry)

    def first_hit(self, x, y, dx, dy):
        """First nonnegative ray distance; infinity means no wall intersection."""
        x, y, dx, dy = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (x, y, dx, dy)))
        hit = np.where(self.geometry.strict_mask(x, y), np.inf, 0.)
        for x1, y1, x2, y2 in self.segments:
            ex, ey = x2 - x1, y2 - y1
            cross = dx * ey - dy * ex
            safe = np.where(cross != 0., cross, 1.)
            distance = ((x1 - x) * ey - (y1 - y) * ex) / safe
            fraction = ((x1 - x) * dy - (y1 - y) * dx) / safe
            valid = (cross != 0.) & (distance >= 0.) & (fraction >= 0.) & (fraction <= 1.)
            hit = np.minimum(hit, np.where(valid, distance, np.inf))
        for cx, cy, a, b, side in self.ellipses:
            xx, yy, vx, vy = (x - cx) / a, (y - cy) / b, dx / a, dy / b
            aa, bb, cc = vx * vx + vy * vy, xx * vx + yy * vy, xx * xx + yy * yy - 1.
            discriminant = bb * bb - aa * cc
            root = np.sqrt(np.maximum(discriminant, 0.))
            q = -bb - np.copysign(root, bb)
            candidates = (q / np.where(aa != 0., aa, 1.), cc / np.where(q != 0., q, 1.))
            for distance in candidates:
                valid = (aa > 0.) & (discriminant >= 0.) & (distance >= 0.)
                if side:
                    valid &= side * (x + distance * dx - cx) >= 0.
                hit = np.minimum(hit, np.where(valid, distance, np.inf))
        return hit

    def cuda_source(self):
        """A device helper to embed in the owning element's fused kernel."""
        source = '#ifndef INFINITY\n#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))\n#endif\n'
        source += _aperture_inside_cuda(self.geometry, 'pass_aperture_inside')
        source += '''
__device__ inline double pass_aperture_hit(
    double x,
    double y,
    double dx,
    double dy
) {
    if (!pass_aperture_inside(x, y))
        return 0.;
    double hit = INFINITY;
'''
        for x1, y1, x2, y2 in self.segments:
            values = ','.join(format(float(v), '.17g') for v in (x1, y1, x2 - x1, y2 - y1))
            source += f'''{{ const double p[4]={{{values}}};
                double cross=dx*p[3]-dy*p[2];
                if (cross!=0.) {{
                    double s=((p[0]-x)*p[3]-(p[1]-y)*p[2])/cross;
                    double t=((p[0]-x)*dy-(p[1]-y)*dx)/cross;
                    if (s>=0. && t>=0. && t<=1.) hit=fmin(hit,s);
                }} }}\n'''
        for cx, cy, a, b, side in self.ellipses:
            values = ','.join(format(float(v), '.17g') for v in (cx, cy, a, b))
            source += f'''{{ const double p[4]={{{values}}};
                double xx=(x-p[0])/p[2],yy=(y-p[1])/p[3],vx=dx/p[2],vy=dy/p[3];
                double aa=vx*vx+vy*vy,bb=xx*vx+yy*vy,cc=xx*xx+yy*yy-1.;
                double disc=bb*bb-aa*cc;
                if (aa>0. && disc>=0.) {{
                    double q=-bb-copysign(sqrt(disc),bb);
                    double roots[2]={{q/aa,q!=0. ? cc/q : 0.}};
                    for (int j=0;j<2;++j) {{ double s=roots[j];
                        if (s>=0. && ({side}==0 || {side}*(x+s*dx-p[0])>=0.)) hit=fmin(hit,s);
                    }}
                }} }}\n'''
        return source + 'return hit; }\n'


def _slab_entry(u, du, lower, upper, tolerance):
    """First contact with a closed u interval, allowing projection roundoff.

    Only an entry contact is snapped to zero distance. Future intersections
    still use the supplied physical surfaces, without expanding the material.
    """
    moving = du != 0.
    safe = np.where(moving, du, 1.)
    a, b = (lower - u) / safe, (upper - u) / safe
    near, far = np.maximum(0., np.minimum(a, b)), np.maximum(a, b)
    inside = (u >= lower - tolerance) & (u <= upper + tolerance)
    return np.where(inside, 0., np.where(moving & (far >= near), near, np.inf))


@dataclass
class _ElectricOrbit:
    """Exact body orbit, with momenta in P0 and energy in P0*c.

    k=q*Eu/(P0*c), ps and pv are constant in the uniform transverse field.
    Edges are handled separately. Stable increments retain the zero-field limit.
    """

    x: float
    y: float
    px: float
    py: float
    dp: float
    beta: float
    inv_g2: float
    k: float
    co: float
    si: float

    def __post_init__(self):
        self.pu = self.px * self.co - self.py * self.si
        self.pv = self.px * self.si + self.py * self.co
        self.ps = math.sqrt((1 + self.dp)**2 - self.px**2 - self.py**2)
        self.energy = math.sqrt(self.inv_g2 + (1 - self.inv_g2) * (1 + self.dp)**2) / self.beta
        self.mass2 = self.inv_g2 / (self.beta * self.beta)
        self.transverse_mass = math.sqrt(self.mass2 + self.ps**2 + self.pv**2)

    def increments(self, distance):
        a = self.k * distance / self.ps
        # shc=sinh(a)/a, cmc=(cosh(a)-1)/a; no division by field strength.
        if abs(a) < 1.e-4:
            a2 = a * a
            shcm1 = a2 * (1 / 6 + a2 * (1 / 120 + a2 / 5040))
            cmc = a * (1 / 2 + a2 * (1 / 24 + a2 * (1 / 720 + a2 / 40320)))
        else:
            shcm1 = math.sinh(a) / a - 1
            cmc = 2 * math.sinh(a / 2)**2 / a
        shc = 1 + shcm1
        du = distance / self.ps * (self.pu * shc + self.energy * cmc)
        dv = distance / self.ps * self.pv
        dpu = a * (self.energy * shc + self.pu * cmc)
        de = a * (self.pu * shc + self.energy * cmc)
        A = self.beta * self.energy
        slip = (self.dp * (2 + self.dp) * self.inv_g2 - self.px**2 - self.py**2) / (self.ps * (self.ps + A))
        dz = distance * slip - self.beta * distance / self.ps * (self.energy * shcm1 + self.pu * cmc)
        return du, dv, dpu, de, dz

    def point(self, distance):
        du, dv, _, _, _ = self.increments(distance)
        return self.x + self.co * du + self.si * dv, self.y - self.si * du + self.co * dv

    def linear_range(self, ax, ay, c, lo, hi):
        """Exact extrema of ax*x(s)+ay*y(s)+c on a forward interval."""
        x0, y0 = self.point(lo)
        x1, y1 = self.point(hi)
        f0, f1 = ax * x0 + ay * y0 + c, ax * x1 + ay * y1 + c
        lower, upper = min(f0, f1), max(f0, f1)
        au, av = ax * self.co - ay * self.si, ax * self.si + ay * self.co
        if au != 0 and self.k != 0:
            target = -av * self.pv / au
            et = math.hypot(self.transverse_mass, target)
            dpu = target - self.pu
            de = dpu * (target + self.pu) / (et + self.energy)
            angle = math.asinh((dpu * self.energy - self.pu * de) / self.transverse_mass**2)
            middle = self.ps / self.k * angle
            if lo < middle < hi:
                x, y = self.point(middle)
                value = ax * x + ay * y + c
                lower, upper = min(lower, value), max(upper, value)
        return lower, upper

    def electrode_contact(self, surface, length):
        """Solve E(u)=E0+k*du before evaluating possibly very long body maps.

        Both momentum roots are considered: a particle may turn before reaching
        a surface. A stable quadratic and rapidity difference retain weak fields.
        """
        du = surface - (self.x * self.co - self.y * self.si)
        de = self.k * du
        if self.energy + de <= 0:
            return math.inf
        change = de * (2 * self.energy + de)
        discriminant = self.pu * self.pu + change
        tolerance = 64 * np.finfo(float).eps * max(self.pu * self.pu, abs(change), 1.e-300)
        if discriminant < -tolerance:
            return math.inf
        root = math.sqrt(max(0., discriminant))
        q = -self.pu - math.copysign(root, self.pu)
        hit = math.inf
        for dpu in (q, -change / q if q else 0.):
            angle = math.asinh((dpu * self.energy - self.pu * de) / self.transverse_mass**2)
            distance = self.ps / self.k * angle
            if 0 <= distance <= length:
                hit = min(hit, distance)
        return hit


def _curve_contact(orbit, length, primitive):
    """Conservative interval isolation of the first contact, not endpoint sampling.

    Coordinate extrema bound each whole interval, including turns/tangencies.
    A root is localized to 1e-12*max(1,length) metres along s. Geometry rounding
    uses double precision independently of particle storage precision.
    """
    tolerance_s = 1.e-12 * max(1., length)
    pending = [(0., length)]
    kind, values = primitive
    while pending:
        lo, hi = pending.pop()
        if kind == 'line':
            ax, ay, c, bx, by, d, extent = values
            f0, f1 = orbit.linear_range(ax, ay, c, lo, hi)
            tolerance = 64 * np.finfo(float).eps * max(1., abs(c), abs(f0), abs(f1))
            if f0 > tolerance or f1 < -tolerance:
                continue
            if math.isfinite(extent):
                t0, t1 = orbit.linear_range(bx, by, d, lo, hi)
                if t1 < -tolerance or t0 > extent + tolerance:
                    continue
        else:
            cx, cy, a, b, side = values
            x0, x1 = orbit.linear_range(1 / a, 0., -cx / a, lo, hi)
            y0, y1 = orbit.linear_range(0., 1 / b, -cy / b, lo, hi)
            if side and (x1 < 0 if side > 0 else x0 > 0):
                continue
            minimum = (0. if x0 <= 0 <= x1 else min(x0 * x0, x1 * x1))
            minimum += 0. if y0 <= 0 <= y1 else min(y0 * y0, y1 * y1)
            maximum = max(x0 * x0, x1 * x1) + max(y0 * y0, y1 * y1)
            tolerance = 64 * np.finfo(float).eps * max(1., maximum)
            if minimum > 1 + tolerance or maximum < 1 - tolerance:
                continue
        if hi - lo <= tolerance_s:
            return (lo + hi) / 2
        middle = (lo + hi) / 2
        pending.append((middle, hi))
        pending.append((lo, middle))
    return math.inf


def _curved_primitives(boundary, co, si, field_start, counter):
    """Boundary equations in the beam frame; electrode planes have no ends."""
    result = [('line', (co, -si, -u, 0., 0., 0., math.inf)) for u in (field_start, counter)]
    for x0, y0, x1, y1 in boundary.segments:
        ex, ey = x1 - x0, y1 - y0
        norm = math.hypot(ex, ey)
        ax, ay, bx, by = -ey / norm, ex / norm, ex / norm, ey / norm
        result.append(('line', (ax, ay, -ax * x0 - ay * y0, bx, by, -bx * x0 - by * y0, norm)))
    result.extend(('ellipse', ellipse) for ellipse in boundary.ellipses)
    return result


@Command.register("elseparator")
class ElSeparator(Command):
    """Signed V or VL, one roll, and an open field region on local +u.

    u=x*cos(tilt)-y*sin(tilt), v=x*sin(tilt)+y*cos(tilt).
    Septum: [position, position+thickness]; counter electrode starts a gap
    beyond that interval. Both plates and the field cover all local v.
    The circulating-beam field-free region is u < position; its surviving
    particles and those in the field region continue into the downstream lattice.
    The separately configured beam-frame vacuum aperture applies everywhere.
    Touching material loses the particle, even with zero voltage. Thick tracking
    includes electric work and ideal hard edges, but not material scattering or
    a measured three-dimensional fringe field. The thin map remains effective.
    """

    def __init__(self, beam_id, sim, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}
        aliases = {field.alias.lower(): field.alias for field in ElSeparatorItem.model_fields.values()}
        unknown = set(kwargs) - set(aliases) - {"name"}
        if unknown:
            raise ValueError("ElSeparator uses V or VL, Gap and Septum position; "
                             "obsolete/unknown parameters: " + ", ".join(sorted(unknown)))
        if kwargs.get("space charge") is not None:
            kwargs["space charge"] = parse_element_space_charge(kwargs["space charge"])
        config = ElSeparatorItem.model_validate({aliases[k]: v for k, v in kwargs.items() if k in aliases})
        self.beam_id, self.cmd_type, self.cmd_name = beam_id, "ElSeparator", kwargs["name"]
        for name in ("s", "length", "voltage", "voltage_length", "gap", "tilt", "septum_position", "septum_thickness", "aperture_type",
                     "aperture_value"):
            setattr(self, name, getattr(config, name))
        self.cos_t, self.sin_t = float(np.cos(self.tilt)), float(np.sin(self.tilt))
        self.field_start = self.septum_position + self.septum_thickness
        self.counter_position = self.field_start + self.gap
        self.integrated_field = (self.voltage_length / self.gap if self.voltage_length is not None else (self.voltage / self.gap) * self.length)
        self.boundary = _ApertureBoundary(build_aperture({"Type": self.aperture_type, "Value": self.aperture_value}))
        self._curved_boundaries = _curved_primitives(self.boundary, self.cos_t, self.sin_t, self.field_start, self.counter_position)
        self._gpu_cache = {}
        configure_element_slicing(self, sim, kwargs)
        super().__init__()

    def print(self):
        logging.getLogger(__name__).info("ElSeparator %s: S=%g L=%g V=%s V VL=%s V*m gap=%g tilt=%g septum=%g thickness=%g aperture=%s %s",
                                         self.cmd_name, self.s, self.length, self.voltage, self.voltage_length, self.gap, self.tilt,
                                         self.septum_position, self.septum_thickness, self.aperture_type, self.aperture_value)
        print_element_slicing(self)

    def _integrated_kick(self, bunch):
        """Full-gap impulse in P_u/P0, using the current reference speed/rigidity."""
        denom = bunch.beta * const.c * bunch.brho
        if not np.isfinite(denom) or denom <= 0 or not 0 < bunch.beta < 1:
            raise ValueError("ElSeparator requires a finite massive-particle reference and positive rigidity")
        value = float(np.sign(bunch.num_charge) * self.integrated_field / denom)
        if not np.isfinite(value):
            raise ValueError("ElSeparator integrated impulse must be finite")
        return value

    def _advance_cpu(self, p, bunch, length, s0, turn, selection=None):
        bunch_slice = slice(bunch.start_idx, bunch.end_idx)
        x, px, y, py, z, dp, tag = (getattr(p, n)[bunch_slice] for n in ("x", "px", "y", "py", "z", "dp", "tag"))
        valid, inv_ps, slip = drift_factors(px, py, dp, 1 / bunch.gamma**2)
        valid &= np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        selected = np.ones(tag.shape, dtype=bool) if selection is None else selection
        invalid = (tag > 0) & selected & ~valid
        tag[invalid] = -np.abs(tag[invalid])
        p.lost_position[bunch_slice][invalid] = s0
        p.lost_turn[bunch_slice][invalid] = turn
        active = (tag > 0) & selected
        xx, yy = np.where(active, x, 0.).astype(float), np.where(active, y, 0.).astype(float)
        dx = np.where(active, px, 0.).astype(float) * inv_ps.astype(float)
        dy = np.where(active, py, 0.).astype(float) * inv_ps.astype(float)
        u = xx * self.cos_t - yy * self.sin_t
        du = dx * self.cos_t - dy * self.sin_t
        # A rotated point on a plate rarely projects to its exact binary u.
        # Include coordinate storage and projection roundoff, for both dtypes.
        tolerance = 4 * np.finfo(p.dtype).eps * (np.abs(xx * self.cos_t) + np.abs(yy * self.sin_t))
        hit = np.minimum(self.boundary.first_hit(xx, yy, dx, dy), _slab_entry(u, du, self.septum_position, self.field_start, tolerance))
        hit = np.minimum(hit, _slab_entry(u, du, self.counter_position, np.inf, tolerance))
        lost = active & (hit <= length)
        distance = np.where(lost, hit, length).astype(p.dtype)
        x[active] += distance[active] * px[active] * inv_ps[active]
        y[active] += distance[active] * py[active] * inv_ps[active]
        z[active] += distance[active] * slip[active]
        tag[lost] = -np.abs(tag[lost])
        p.lost_position[bunch_slice][lost] = s0 + hit[lost]
        p.lost_turn[bunch_slice][lost] = turn

    def _kick_cpu(self, p, bunch, kick):
        bunch_slice = slice(bunch.start_idx, bunch.end_idx)
        x, y, tag = p.x[bunch_slice], p.y[bunch_slice], p.tag[bunch_slice]
        u = x.astype(float) * self.cos_t - y.astype(float) * self.sin_t
        mask = (tag > 0) & (u > self.field_start) & (u < self.counter_position)
        px, py, dp = (getattr(p, n)[bunch_slice][mask].astype(float) for n in ('px', 'py', 'dp'))
        ps = np.sqrt((1 + dp)**2 - px * px - py * py)
        A = np.sqrt(1 / bunch.gamma**2 + (1 - 1 / bunch.gamma**2) * (1 + dp)**2)
        impulse = kick * A / ps
        p.px[bunch_slice][mask] += impulse * self.cos_t
        p.py[bunch_slice][mask] -= impulse * self.sin_t

    def _edge_cpu(self, p, bunch, k, entering, s0, turn):
        """Match mechanical energy across Phi=-E*(u-field_start), in zero time."""
        bunch_slice = slice(bunch.start_idx, bunch.end_idx)
        x, px, y, py, dp, tag = (getattr(p, n)[bunch_slice] for n in ('x', 'px', 'y', 'py', 'dp', 'tag'))
        u = x.astype(float) * self.cos_t - y.astype(float) * self.sin_t
        indices = np.flatnonzero((tag > 0) & (u > self.field_start) & (u < self.counter_position))
        if not len(indices) or k == 0:
            return
        old = dp[indices].astype(float)
        inv_g2 = 1 / bunch.gamma**2
        energy = np.sqrt(inv_g2 + (1 - inv_g2) * (1 + old)**2) / bunch.beta
        de = (1 if entering else -1) * k * (u[indices] - self.field_start)
        delta_r2 = de * (2 * energy + de)
        r2 = (1 + old)**2 + delta_r2
        transverse = px[indices].astype(float)**2 + py[indices].astype(float)**2
        valid = np.isfinite(r2) & np.isfinite(energy + de) & (energy + de > 0) & (r2 > transverse)
        lost = indices[~valid]
        tag[lost] = -np.abs(tag[lost])
        p.lost_position[bunch_slice][lost], p.lost_turn[bunch_slice][lost] = s0, turn
        good = indices[valid]
        value = (old[valid] * (2 + old[valid]) + delta_r2[valid]) / (np.sqrt(r2[valid]) + 1)
        dp[good] = value

    def _body_cpu(self, p, bunch, k, length, s0, turn):
        self._advance_cpu(p, bunch, 0., s0, turn)
        bunch_slice = slice(bunch.start_idx, bunch.end_idx)
        u = p.x[bunch_slice].astype(float) * self.cos_t - p.y[bunch_slice].astype(float) * self.sin_t
        field = (p.tag[bunch_slice] > 0) & (u > self.field_start) & (u < self.counter_position) & (k != 0.)
        self._advance_cpu(p, bunch, length, s0, turn, selection=~field)
        for local in np.flatnonzero(field):
            i = bunch.start_idx + local
            orbit = _ElectricOrbit(*(float(getattr(p, n)[i]) for n in ('x', 'y', 'px', 'py', 'dp')), bunch.beta, 1 / bunch.gamma**2, k, self.cos_t,
                                   self.sin_t)
            hit = min(orbit.electrode_contact(self.field_start, length), orbit.electrode_contact(self.counter_position, length))
            for primitive in self._curved_boundaries[2:]:
                hit = min(hit, _curve_contact(orbit, min(length, hit), primitive))
            distance = min(length, hit)
            du, dv, dpu, de, dz = orbit.increments(distance)
            p.x[i] += self.cos_t * du + self.sin_t * dv
            p.y[i] += -self.sin_t * du + self.cos_t * dv
            p.px[i] += self.cos_t * dpu
            p.py[i] -= self.sin_t * dpu
            r2 = (1 + orbit.dp)**2 + de * (2 * orbit.energy + de)
            p.dp[i] = (orbit.dp * (2 + orbit.dp) + de * (2 * orbit.energy + de)) / (math.sqrt(r2) + 1)
            p.z[i] += dz
            if hit <= length:
                p.tag[i] = -abs(p.tag[i])
                p.lost_position[i], p.lost_turn[i] = s0 + hit, turn
        # Rounded storage at a body/SC boundary must not feed a contact or an
        # invalid forward state into the collective solver before the next step.
        self._advance_cpu(p, bunch, 0., s0 + length, turn)

    def _kernel(self, p):
        import cupy as cp
        key = (cp.cuda.runtime.getDevice(), np.dtype(p.dtype))
        if key not in self._gpu_cache:
            source = _kernel_source(self.boundary.cuda_source())
            self._gpu_cache[key] = cp.RawKernel(source,
                                                "track_separator",
                                                options=("--std=c++17", "--fmad=false", f"-DPASS_USE_FLOAT={int(p.dtype==np.float32)}"))
        return self._gpu_cache[key]

    def _thick_kernel(self, p):
        import cupy as cp
        key = ('thick', cp.cuda.runtime.getDevice(), np.dtype(p.dtype))
        if key not in self._gpu_cache:
            source = _kernel_source(self.boundary.cuda_source()) + _thick_kernel_source(self._curved_boundaries)
            self._gpu_cache[key] = cp.RawKernel(source,
                                                'track_separator_thick',
                                                options=('--std=c++17', '--fmad=false', f'-DPASS_USE_FLOAT={int(p.dtype==np.float32)}'))
        return self._gpu_cache[key]

    def execute_cpu(self, sim):
        return self._execute(sim, gpu=False)

    def execute_gpu(self, sim):
        return self._execute(sim, gpu=True)

    def _execute(self, sim, *, gpu):
        beam, turn = sim.beams[self.beam_id], int(sim.state.turn)
        p = beam.particles
        kernel = self._kernel(p) if gpu and self.length == 0 else None
        thick_kernel = self._thick_kernel(p) if gpu and self.length > 0 else None
        for bunch in beam.bunches:
            total_kick = self._integrated_kick(bunch)
            field_k = (np.sign(bunch.num_charge) * (self.integrated_field / self.length) / (const.c * bunch.brho) if self.length > 0 else 0.)
            if not np.isfinite(field_k):
                raise ValueError("ElSeparator normalized body field must be finite")
            offset = 0.

            def advance(length):
                nonlocal offset
                self._advance_cpu(p, bunch, length, self.s - self.length + offset, turn)
                offset += length
                bunch.t0 += length / (bunch.beta * const.c)

            def launch(before, kick, after, repeats=1):
                nonlocal offset
                n = bunch.end_idx - bunch.start_idx
                if n:
                    kernel(((n + 255) // 256, ), (256, ),
                           (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(bunch.start_idx), np.int32(
                               bunch.end_idx), np.int32(turn), np.int32(repeats), np.float64(before), p.real(kick), np.float64(after),
                            np.float64(self.s - self.length + offset), p.real(1 / bunch.gamma**2), np.float64(self.cos_t), np.float64(
                                self.sin_t), np.float64(self.septum_position), np.float64(self.field_start), np.float64(self.counter_position)))
                for _ in range(repeats):
                    offset += before
                    bunch.t0 += before / (bunch.beta * const.c)
                    offset += after
                    bunch.t0 += after / (bunch.beta * const.c)

            def thick(action, ds=0.):
                nonlocal offset
                k = field_k
                s0 = self.s - self.length + offset
                if gpu:
                    n = bunch.end_idx - bunch.start_idx
                    if n:
                        thick_kernel(
                            ((n + 255) // 256, ), (256, ),
                            (p.x, p.px, p.y, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(bunch.start_idx), np.int32(
                                bunch.end_idx), np.int32(turn), np.int32(action), np.float64(ds), np.float64(k), np.float64(s0), np.float64(
                                    bunch.beta), np.float64(1 / bunch.gamma**2), np.float64(self.cos_t), np.float64(self.sin_t),
                             np.float64(self.septum_position), np.float64(self.field_start), np.float64(self.counter_position)))
                else:
                    if action == 1:
                        self._body_cpu(p, bunch, k, ds, s0, turn)
                    else:
                        self._advance_cpu(p, bunch, 0., s0, turn)
                        self._edge_cpu(p, bunch, k, action == 0, s0, turn)
                        self._advance_cpu(p, bunch, 0., s0, turn)
                if action == 1:
                    offset += ds
                    bunch.t0 += ds / (bunch.beta * const.c)

            def transport(ds, on_center):
                if on_center is None:
                    thick(1, ds)
                else:
                    thick(1, ds / 2)
                    on_center()
                    thick(1, ds / 2)

            if self.length == 0:
                if gpu:
                    launch(0., total_kick, 0.)
                else:
                    advance(0.)
                    self._kick_cpu(p, bunch, total_kick)
                    advance(0.)  # Reject invalid post-kick momentum at the same plane.
            else:
                thick(0)
                if self._sc_nodes:
                    run_body_slices(self, beam, bunch, turn, transport, gpu=gpu)
                else:
                    thick(1, self.length)
                thick(2)
        return True


def _kernel_source(aperture_source):
    return r'''
#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif
using R = pass_real_t;
''' + drift_cuda_factors() + aperture_source + r'''
__device__ inline double slab_entry(
    double u,
    double du,
    double lo,
    double hi,
    double tolerance
) {
    if (u >= lo - tolerance && u <= hi + tolerance)
        return 0.;
    if (du == 0.)
        return INFINITY;
    double a = (lo - u) / du, b = (hi - u) / du;
    double near = fmax(0., fmin(a, b)), far = fmax(a, b);
    return far >= near ? near : INFINITY;
}
__device__ inline bool separator_advance(
    R& x,
    R& y,
    R& z,
    R px,
    R py,
    R dp,
    int& tag,
    float& lp,
    int& lt,
    R inv_g2,
    double length,
    double s0,
    int turn,
    double co,
    double si,
    double us,
    double outer,
    double counter
) {
    if (tag <= 0)
        return false;
    R inv_ps, slip;
    if (!pass_drift_factors(px, py, dp, inv_g2, inv_ps, slip) || !isfinite(x) || !isfinite(y) || !isfinite(z)) {
        tag = -abs(tag);
        lp = (float)s0;
        lt = turn;
        return false;
    }
    double dx = (double)px * (double)inv_ps, dy = (double)py * (double)inv_ps;
    double u = (double)x * co - (double)y * si;
    double du = dx * co - dy * si;
    const double epsilon = PASS_USE_FLOAT ? 0x1p-23 : 0x1p-52;
    double tolerance = 4. * epsilon * (fabs((double)x * co) + fabs((double)y * si));
    double hit = pass_aperture_hit((double)x, (double)y, dx, dy);
    hit = fmin(hit, slab_entry(u, du, us, outer, tolerance));
    hit = fmin(hit, slab_entry(u, du, counter, INFINITY, tolerance));
    bool lost = hit <= length;
    R distance = (R)(lost ? hit : length);
    x += distance * px * inv_ps;
    y += distance * py * inv_ps;
    z += distance * slip;
    if (lost) {
        tag = -abs(tag);
        lp = (float)(s0 + hit);
        lt = turn;
        return false;
    }
    return true;
}
extern "C" __global__ void track_separator(
    R* x,
    R* px,
    R* y,
    R* py,
    R* z,
    const R* dp,
    int* tag,
    float* lost_position,
    int* lost_turn,
    int start,
    int end,
    int turn,
    int repeats,
    double before,
    R kick,
    double after,
    double s0,
    R inv_g2,
    double co,
    double si,
    double us,
    double outer,
    double counter
) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end || tag[i] <= 0)
        return;
    R xi = x[i], yi = y[i], zi = z[i], pxi = px[i], pyi = py[i], dpi = dp[i];
    int ti = tag[i], lt = lost_turn[i];
    float lp = lost_position[i];
    for (int j = 0; j < repeats; ++j) {
        if (!separator_advance(xi, yi, zi, pxi, pyi, dpi, ti, lp, lt, inv_g2, before, s0, turn, co, si, us, outer, counter))
            break;
        s0 += before;
        double u = (double)xi * co - (double)yi * si;
        if (u > outer && u < counter) {
            double r = 1. + (double)dpi;
            double ps = sqrt(r * r - (double)pxi * pxi - (double)pyi * pyi);
            double A = sqrt((double)inv_g2 + (1. - (double)inv_g2) * r * r);
            double impulse = (double)kick * A / ps;
            pxi += (R)(impulse * co);
            pyi -= (R)(impulse * si);
        }
        if (!separator_advance(xi, yi, zi, pxi, pyi, dpi, ti, lp, lt, inv_g2, after, s0, turn, co, si, us, outer, counter))
            break;
        s0 += after;
    }
    x[i] = xi;
    y[i] = yi;
    z[i] = zi;
    px[i] = pxi;
    py[i] = pyi;
    tag[i] = ti;
    lost_position[i] = lp;
    lost_turn[i] = lt;
}
'''


def _thick_kernel_source(primitives):
    """Same analytic map and conservative interval bounds as the CPU path."""
    source = r'''
struct ElectricOrbit {
    double x, y, px, py, dp, beta, ig, k, co, si, pu, pv, ps, energy, mt;
    __device__ void init(
        double X,
        double Y,
        double PX,
        double PY,
        double DP,
        double B,
        double IG,
        double K,
        double CO,
        double SI
    ) {
        x = X;
        y = Y;
        px = PX;
        py = PY;
        dp = DP;
        beta = B;
        ig = IG;
        k = K;
        co = CO;
        si = SI;
        pu = px * co - py * si;
        pv = px * si + py * co;
        ps = sqrt((1 + dp) * (1 + dp) - px * px - py * py);
        energy = sqrt(ig + (1 - ig) * (1 + dp) * (1 + dp)) / beta;
        mt = sqrt(ig / (beta * beta) + ps * ps + pv * pv);
    }
    __device__ void increments(
        double ds,
        double& du,
        double& dv,
        double& dpu,
        double& de,
        double& dz
    ) const {
        double a = k * ds / ps, shcm1, cmc;
        if (fabs(a) < 1.e-4) {
            double a2 = a * a;
            shcm1 = a2 * (1. / 6 + a2 * (1. / 120 + a2 / 5040));
            cmc = a * (1. / 2 + a2 * (1. / 24 + a2 * (1. / 720 + a2 / 40320)));
        } else {
            shcm1 = sinh(a) / a - 1;
            cmc = 2 * sinh(a / 2) * sinh(a / 2) / a;
        }
        double shc = 1 + shcm1;
        du = ds / ps * (pu * shc + energy * cmc);
        dv = ds / ps * pv;
        dpu = a * (energy * shc + pu * cmc);
        de = a * (pu * shc + energy * cmc);
        double A = beta * energy, slip = (dp * (2 + dp) * ig - px * px - py * py) / (ps * (ps + A));
        dz = ds * slip - beta * ds / ps * (energy * shcm1 + pu * cmc);
    }
    __device__ void point(
        double ds,
        double& X,
        double& Y
    ) const {
        double du, dv, dpu, de, dz;
        increments(ds, du, dv, dpu, de, dz);
        X = x + co * du + si * dv;
        Y = y - si * du + co * dv;
    }
    __device__ void range(
        double ax,
        double ay,
        double c,
        double lo,
        double hi,
        double& lower,
        double& upper
    ) const {
        double x0, y0, x1, y1;
        point(lo, x0, y0);
        point(hi, x1, y1);
        double f0 = ax * x0 + ay * y0 + c, f1 = ax * x1 + ay * y1 + c;
        lower = fmin(f0, f1);
        upper = fmax(f0, f1);
        double au = ax * co - ay * si, av = ax * si + ay * co;
        if (au != 0 && k != 0) {
            double target = -av * pv / au, et = hypot(mt, target), dpu = target - pu;
            double de = dpu * (target + pu) / (et + energy);
            double angle = asinh((dpu * energy - pu * de) / (mt * mt)), middle = ps / k * angle;
            if (lo < middle && middle < hi) {
                double X, Y;
                point(middle, X, Y);
                double value = ax * X + ay * Y + c;
                lower = fmin(lower, value);
                upper = fmax(upper, value);
            }
        }
    }
    __device__ double electrode(
        double surface,
        double length
    ) const {
        double de = k * (surface - (x * co - y * si));
        if (energy + de <= 0)
            return INFINITY;
        double change = de * (2 * energy + de), disc = pu * pu + change;
        double tol = 64 * 0x1p-52 * fmax(fmax(pu * pu, fabs(change)), 1.e-300);
        if (disc < -tol)
            return INFINITY;
        double root = sqrt(fmax(0., disc)), q = -pu - copysign(root, pu), hit = INFINITY;
        double increments[2] = {q, q != 0 ? -change / q : 0.};
        for (int j = 0; j < 2; ++j) {
            double angle = asinh((increments[j] * energy - pu * de) / (mt * mt)), ds = ps / k * angle;
            if (ds >= 0 && ds <= length)
                hit = fmin(hit, ds);
        }
        return hit;
    }
};
__device__ double curve_contact(
    const ElectricOrbit& o,
    double length,
    int kind,
    const double* p
) {
    const double eps = 0x1p-52, stol = 1.e-12 * fmax(1., length);
    double left[64], right[64];
    int count = 1;
    left[0] = 0;
    right[0] = length;
    while (count) {
        --count;
        double lo = left[count], hi = right[count];
        if (kind == 0) {
            double f0, f1;
            o.range(p[0], p[1], p[2], lo, hi, f0, f1);
            double tol = 64 * eps * fmax(1., fmax(fabs(p[2]), fmax(fabs(f0), fabs(f1))));
            if (f0 > tol || f1 < -tol)
                continue;
            if (isfinite(p[6])) {
                double t0, t1;
                o.range(p[3], p[4], p[5], lo, hi, t0, t1);
                if (t1 < -tol || t0 > p[6] + tol)
                    continue;
            }
        } else {
            double x0, x1, y0, y1;
            o.range(1 / p[2], 0, -p[0] / p[2], lo, hi, x0, x1);
            o.range(0, 1 / p[3], -p[1] / p[3], lo, hi, y0, y1);
            if ((p[4] > 0 && x1 < 0) || (p[4] < 0 && x0 > 0))
                continue;
            double minimum = (x0 <= 0 && x1 >= 0) ? 0 : fmin(x0 * x0, x1 * x1);
            minimum += (y0 <= 0 && y1 >= 0) ? 0 : fmin(y0 * y0, y1 * y1);
            double maximum = fmax(x0 * x0, x1 * x1) + fmax(y0 * y0, y1 * y1);
            double tol = 64 * eps * fmax(1., maximum);
            if (minimum > 1 + tol || maximum < 1 - tol)
                continue;
        }
        if (hi - lo <= stol)
            return (lo + hi) / 2;
        double mid = (lo + hi) / 2;
        left[count] = mid;
        right[count] = hi;
        ++count;
        left[count] = lo;
        right[count] = mid;
        ++count;
    }
    return INFINITY;
}
__device__ double curved_hit(
    const ElectricOrbit& o,
    double length
) {
    double hit = INFINITY;
'''
    for _, values in primitives[:2]:
        surface = -values[2]
        source += f'hit=fmin(hit,o.electrode({format(surface,".17g")},length));\n'
    for kind, values in primitives[2:]:
        numbers = ','.join('INFINITY' if math.isinf(v) else format(float(v), '.17g') for v in values)
        source += f'{{const double p[]={{{numbers}}}; hit=fmin(hit,curve_contact(o,fmin(length,hit),{int(kind!="line")},p));}}\n'
    return source + r'''
return hit;
}
extern "C" __global__ void track_separator_thick(
    R* x,
    R* px,
    R* y,
    R* py,
    R* z,
    R* dp,
    int* tag,
    float* lp,
    int* lt,
    int start,
    int end,
    int turn,
    int action,
    double length,
    double k,
    double s0,
    double beta,
    double ig,
    double co,
    double si,
    double us,
    double outer,
    double counter
) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end || tag[i] <= 0)
        return;
    R X = x[i], Y = y[i], Z = z[i], PX = px[i], PY = py[i], DP = dp[i];
    int T = tag[i], LT = lt[i];
    float LP = lp[i];
    if (separator_advance(X, Y, Z, PX, PY, DP, T, LP, LT, (R)ig, 0, s0, turn, co, si, us, outer, counter)) {
        double u = (double)X * co - (double)Y * si;
        bool field = u > outer && u < counter && k != 0;
        if (action != 1 && field) {
            double old = (double)DP, energy = sqrt(ig + (1 - ig) * (1 + old) * (1 + old)) / beta;
            double de = (action == 0 ? 1. : -1.) * k * (u - outer), dr2 = de * (2 * energy + de);
            double r2 = (1 + old) * (1 + old) + dr2, transverse = (double)PX * PX + (double)PY * PY;
            if (!isfinite(r2) || !isfinite(energy + de) || energy + de <= 0 || r2 <= transverse) {
                T = -abs(T);
                LP = (float)s0;
                LT = turn;
            } else
                DP = (R)((old * (2 + old) + dr2) / (sqrt(r2) + 1));
            separator_advance(X, Y, Z, PX, PY, DP, T, LP, LT, (R)ig, 0, s0, turn, co, si, us, outer, counter);
        } else if (action == 1 && !field) {
            separator_advance(X, Y, Z, PX, PY, DP, T, LP, LT, (R)ig, length, s0, turn, co, si, us, outer, counter);
        } else if (action == 1) {
            ElectricOrbit o;
            o.init(X, Y, PX, PY, DP, beta, ig, k, co, si);
            double hit = curved_hit(o, length), ds = fmin(length, hit), du, dv, dpu, de, dz;
            o.increments(ds, du, dv, dpu, de, dz);
            X += (R)(co * du + si * dv);
            Y += (R)(-si * du + co * dv);
            PX += (R)(co * dpu);
            PY -= (R)(si * dpu);
            Z += (R)dz;
            double dr2 = de * (2 * o.energy + de), r2 = (1 + o.dp) * (1 + o.dp) + dr2;
            DP = (R)((o.dp * (2 + o.dp) + dr2) / (sqrt(r2) + 1));
            if (hit <= length) {
                T = -abs(T);
                LP = (float)(s0 + hit);
                LT = turn;
            }
        }
        if (action == 1 && T > 0)
            separator_advance(X, Y, Z, PX, PY, DP, T, LP, LT, (R)ig, 0, s0 + length, turn, co, si, us, outer, counter);
    }
    x[i] = X;
    y[i] = Y;
    z[i] = Z;
    px[i] = PX;
    py[i] = PY;
    dp[i] = DP;
    tag[i] = T;
    lp[i] = LP;
    lt[i] = LT;
}
'''
