"""Explicit paired crossing charts; the collision frame stores eta in dp.

At zero angle ring A uses laboratory (X,Y,Z), ring B uses (-X,Y,-Z).
The collision frame uses common transverse axes and positive-early z on
both sides. The reference momentum inside the chart is P0*cos(Theta/2).
"""

import logging

import numpy as np

from PASS.commands.command import Command
from PASS.para.schema.elements import CrossingAngleItem
from PASS.utils.coordinates import delta_to_eta, eta_to_delta


def transform_collision_coordinates(state, full_angle, plane, side, inverse=False, xp=np):
    """Transform a canonical six-tuple; no particle pools or clocks are changed."""
    if side not in (0, 1) or not np.isfinite(full_angle) or abs(full_angle) >= np.pi or not np.isfinite(plane):
        raise ValueError('Crossing geometry requires side 0/1 and finite abs(full angle)<pi')
    direction = 1 if side == 0 else -1
    phi, alpha = direction * full_angle / 2, direction * plane
    sn, cs, tn = float(np.sin(phi)), float(np.cos(phi)), float(np.tan(phi))
    sa, ca = float(np.sin(alpha)), float(np.cos(alpha))
    x, px, y, py, z, eta = state
    if not inverse:
        x, y, px, py = ca * x + sa * y, -sa * x + ca * y, ca * px + sa * py, -sa * px + ca * py
        ps = xp.sqrt((1 + eta)**2 - px * px - py * py)
        h = (px * px + py * py) / (1 + eta + ps)
        px_star, py_star = (px - h * tn) / cs, py / cs
        eta_star = eta - px * tn + h * tn * tn
        ps_star = xp.sqrt((1 + eta_star)**2 - px_star * px_star - py_star * py_star)
        h_star = (px_star * px_star + py_star * py_star) / (1 + eta_star + ps_star)
        return (direction * (z * tn + (1 + px_star / ps_star * sn) * x), direction * px_star, y + py_star / ps_star * sn * x, py_star,
                z / cs - h_star / ps_star * sn * x, eta_star)
    x, px = direction * x, direction * px
    ps = xp.sqrt((1 + eta)**2 - px * px - py * py)
    h = (px * px + py * py) / (1 + eta + ps)
    u = (x - z * sn) / (1 + px / ps * sn + h / ps * sn * sn)
    v = y - py / ps * sn * u
    z_new = z * cs + h / ps * sn * cs * u
    pu, pv, eta_new = cs * px + cs * sn * h, cs * py, eta + sn * px
    return ca * u - sa * v, ca * pu - sa * pv, sa * u + ca * v, sa * pu + ca * pv, z_new, eta_new


@Command.register('crossingangle')
class CrossingAngle(Command):

    def __init__(self, beam_id, sim, **command_kwargs):
        parameters = CrossingAngleItem.model_validate(command_kwargs)
        self.beam_id, self.s, self.order = beam_id, parameters.s, parameters.order
        self.cmd_name, self.cmd_type, self.length = parameters.name, 'CrossingAngle', 0.
        self.configuration, self.direction = parameters.configuration, parameters.direction
        self.is_enabled, self._gpu_cache = True, {}

    def print(self):
        logging.getLogger(__name__).info('S=%g, Command=%s, Name=%s, Configuration=%s, Direction=%s', self.s, self.cmd_type, self.cmd_name,
                                         self.configuration, self.direction)

    def execute_cpu(self, sim):
        return self._execute(sim)

    def execute_gpu(self, sim):
        return self._execute(sim)

    def _execute(self, sim):
        from PASS.commands.beam_beam import get_collision_coordinator

        if not get_collision_coordinator(sim).is_enabled(self.configuration):
            return False
        configuration = sim.cfg.beam_beam_configurations[self.configuration]
        if self.beam_id not in configuration.beams:
            raise ValueError(f'Beam {self.beam_id} is absent from collision configuration {self.configuration}')
        self.side_index = configuration.beams.index(self.beam_id)
        self.full_angle, self.plane = configuration.full_crossing_angle, configuration.crossing_plane
        beam = sim.beams[self.beam_id]
        inverse = self.direction == 'inverse'
        for bunch in beam.bunches:
            frame = getattr(bunch, 'collision_frame', None)
            if inverse:
                if frame is None or frame.get('configuration') != self.configuration or frame.get('frame') != 'collision':
                    raise ValueError(f'CrossingAngle inverse for {self.configuration} has no matching collision frame')
                if (frame['particle_range'] != (bunch.start_idx, bunch.end_idx) or frame['reference_p0'] != bunch.p0 or frame['beta'] != bunch.beta
                        or frame['reference_time'] != bunch.t0):
                    raise ValueError('CrossingAngle inverse requires unchanged bunch range and public reference state')
            elif frame is not None:
                raise ValueError('CrossingAngle cannot enter an already transformed bunch')
            if not 0 < bunch.beta <= 1:
                raise ValueError('CrossingAngle requires 0 < reference beta <= 1')
            direction = 1 if self.side_index == 0 else -1
            if not inverse:
                frame = {
                    'configuration': self.configuration,
                    'frame': 'transforming',
                    'p0': bunch.p0 * np.cos(self.full_angle / 2),
                    'beta': bunch.beta,
                    'phi': direction * self.full_angle / 2,
                    'crossing_plane': direction * self.plane,
                    'side_index': self.side_index,
                    'particle_range': (bunch.start_idx, bunch.end_idx),
                    'reference_p0': bunch.p0,
                    'reference_time': bunch.t0
                }
                bunch.collision_frame = frame
            try:
                if beam.particles.xp is np:
                    self._track_cpu(beam.particles, bunch, inverse)
                else:
                    self._track_gpu(beam.particles, bunch, inverse)
            except Exception:
                frame['frame'] = 'error'
                raise
            bunch.collision_frame = None if inverse else frame
            if not inverse:
                frame['frame'] = 'collision'
        return True

    def _track_cpu(self, p, bunch, inverse):
        # Bounded temporary chunks preserve storage precision and pool ownership.
        beta = float(bunch.beta)
        for start in range(bunch.start_idx, bunch.end_idx, 16384):
            end = min(start + 16384, bunch.end_idx)
            active = np.flatnonzero(p.tag[start:end] > 0) + start
            state = tuple(getattr(p, name)[active] for name in ('x', 'px', 'y', 'py', 'z', 'dp'))
            if not inverse:
                valid = state[5] > -1
                valid &= (1 + state[5])**2 > state[1]**2 + state[3]**2
                if not np.all(valid):
                    raise ValueError('CrossingAngle needs positive forward mechanical momentum')
                state = (*state[:5], delta_to_eta(state[5], beta))
            with np.errstate(invalid='ignore', divide='ignore'):
                result = transform_collision_coordinates(state, self.full_angle, self.plane, self.side_index, inverse)
            eta = result[5]
            if inverse:
                delta = eta_to_delta(eta, beta)
                valid = (1 + beta**2 * eta > 0) & (1 + delta > 0) & ((1 + delta)**2 > result[1]**2 + result[3]**2)
                result = (*result[:5], delta)
            else:
                valid = (1 + eta > 0) & ((1 + eta)**2 > result[1]**2 + result[3]**2)
            if not np.all(valid) or not all(np.all(np.isfinite(value)) for value in result):
                raise ValueError('CrossingAngle encountered nonfinite or non-forward coordinates')
            for name, value in zip(('x', 'px', 'y', 'py', 'z', 'dp'), result):
                getattr(p, name)[active] = value

    def _track_gpu(self, p, bunch, inverse):
        import cupy as cp

        if bunch.start_idx == bunch.end_idx:
            return
        if 'crossing' not in self._gpu_cache:
            self._gpu_cache['crossing'] = cp.ElementwiseKernel(
                'int32 tag, T beta, T rest_fraction, T phi, T alpha, int32 direction, int32 inverse',
                'T x, T px, T y, T py, T z, T longitudinal, int32 invalid', r'''
invalid = 0;
if (tag > 0) {
    T u = x, pu = px, v = y, pv = py, zz = z, eta = longitudinal;
    T sn = sin(phi), cs = cos(phi), tn = tan(phi), sa = sin(alpha), ca = cos(alpha);
    bool valid = isfinite(u) && isfinite(pu) && isfinite(v) && isfinite(pv) && isfinite(zz) && isfinite(eta);
    if (!inverse) {
        valid = valid && eta > -1 && (1 + eta) * (1 + eta) > pu * pu + pv * pv;
        T value = eta * (2 + eta);
        T momentum_ratio = 1 + eta;
        eta = value / (sqrt(rest_fraction + beta * beta * momentum_ratio * momentum_ratio) + 1);
        T old_u = u, old_pu = pu;
        u = ca * old_u + sa * v;
        v = -sa * old_u + ca * v;
        pu = ca * old_pu + sa * pv;
        pv = -sa * old_pu + ca * pv;
        T ps = sqrt((1 + eta) * (1 + eta) - pu * pu - pv * pv);
        T h = (pu * pu + pv * pv) / (1 + eta + ps);
        T pu_star = (pu - h * tn) / cs, pv_star = pv / cs;
        eta = eta - pu * tn + h * tn * tn;
        T ps_star = sqrt((1 + eta) * (1 + eta) - pu_star * pu_star - pv_star * pv_star);
        T h_star = (pu_star * pu_star + pv_star * pv_star) / (1 + eta + ps_star);
        valid = valid && ps > 0 && ps_star > 0 && 1 + eta > 0;
        T u_star = zz * tn + (1 + pu_star / ps_star * sn) * u;
        v += pv_star / ps_star * sn * u;
        zz = zz / cs - h_star / ps_star * sn * u;
        u = direction * u_star;
        pu = direction * pu_star;
        pv = pv_star;
    } else {
        u *= direction;
        pu *= direction;
        T ps = sqrt((1 + eta) * (1 + eta) - pu * pu - pv * pv);
        T h = (pu * pu + pv * pv) / (1 + eta + ps);
        T denominator = 1 + pu / ps * sn + h / ps * sn * sn;
        valid = valid && ps > 0 && denominator != 0 && 1 + eta > 0;
        u = (u - zz * sn) / denominator;
        v -= pv / ps * sn * u;
        zz = zz * cs + h / ps * sn * cs * u;
        eta += sn * pu;
        pu = cs * pu + cs * sn * h;
        pv *= cs;
        T value = 2 * eta + beta * beta * eta * eta;
        T squared_ratio = (1 + 2 * eta) + beta * beta * eta * eta;
        if (eta < 0 && beta > T(0.5)) {
            squared_ratio = (1 + eta) * (1 + eta) - rest_fraction * eta * eta;
        }
        T delta = value / (sqrt(squared_ratio) + 1);
        valid = valid && 1 + beta * beta * eta > 0 && 1 + delta > 0 && (1 + delta) * (1 + delta) > pu * pu + pv * pv;
        eta = delta;
        T old_u = u, old_pu = pu;
        u = ca * old_u - sa * v;
        v = sa * old_u + ca * v;
        pu = ca * old_pu - sa * pv;
        pv = sa * old_pu + ca * pv;
    }
    valid = valid && isfinite(u) && isfinite(pu) && isfinite(v) && isfinite(pv) && isfinite(zz) && isfinite(eta);
    if (valid) {
        x = u;
        px = pu;
        y = v;
        py = pv;
        z = zz;
        longitudinal = eta;
    } else {
        invalid = 1;
    }
}
''', 'pass_crossing_angle')
        selection = slice(bunch.start_idx, bunch.end_idx)
        direction = 1 if self.side_index == 0 else -1
        real = p.dtype.type
        status_key = ('invalid', bunch.end_idx - bunch.start_idx)
        if status_key not in self._gpu_cache:
            self._gpu_cache[status_key] = cp.empty(bunch.end_idx - bunch.start_idx, dtype=cp.int32)
        invalid = self._gpu_cache[status_key]
        plane = float(np.arctan2(np.sin(self.plane), np.cos(self.plane)))
        self._gpu_cache['crossing'](p.tag[selection], real(bunch.beta), real((1 - bunch.beta) * (1 + bunch.beta)),
                                    real(direction * self.full_angle / 2), real(direction * plane), direction, int(inverse),
                                    *(getattr(p, name)[selection] for name in ('x', 'px', 'y', 'py', 'z', 'dp')), invalid)
        if bool(cp.any(invalid)):
            raise ValueError('CrossingAngle encountered invalid coordinates; frame state is incomplete')
