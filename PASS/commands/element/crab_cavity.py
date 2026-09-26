"""Crab cavity and shared canonical IP-equivalent optics for thin elements."""

import logging

import numpy as np

from PASS.commands.command import Command
from PASS.para.schema.elements import CrabCavityItem
from PASS.utils.constants import const
from PASS.utils.coordinates import delta_to_eta, eta_to_delta
from PASS.utils.program import LinearProgram


def resolve_ip_optics(sim, beam_id, reference, expected_s=None):
    """Read the named Twiss endpoint from the owning beam's configuration."""
    matches = [value for name, value in sim.cfg.input_data[beam_id]['sequence'].items() if name.lower() == reference.lower()]
    if len(matches) != 1:
        raise ValueError(f'Optics reference {reference!r} must identify one Twiss node in beam {beam_id}')
    values = {key.lower(): value for key, value in matches[0].items()}
    if str(values.get('command', '')).lower() != 'twiss':
        raise ValueError(f'Optics reference {reference!r} does not identify a Twiss command')
    if expected_s is not None and abs(float(values.get('s (m)', np.inf)) - expected_s) > const.eps:
        raise ValueError(f'Optics reference {reference!r} must be at the same IP position as the equivalent element')
    if float(values.get('dy (m)', 0.)) != 0. or float(values.get('dpy', 0.)) != 0.:
        raise ValueError('IP Twiss transport currently supports horizontal dispersion only; vertical equivalent-endpoint dispersion is supported')
    aliases = {
        'beta_x': 'beta x (m)',
        'alpha_x': 'alpha x',
        'beta_y': 'beta y (m)',
        'alpha_y': 'alpha y',
        'dx': 'dx (m)',
        'dpx': 'dpx',
        'dy': 'dy (m)',
        'dpy': 'dpy'
    }
    optics = {name: float(values.get(alias, 0.)) for name, alias in aliases.items()}
    if optics['beta_x'] <= 0 or optics['beta_y'] <= 0 or not all(np.isfinite(value) for value in optics.values()):
        raise ValueError('IP optics require finite values and positive transverse beta functions')
    return optics


def _dispersion_lift(state, dispersion, beta0, inverse=False, xp=np):
    x, px, y, py, z, eta = state
    dx, dpx, dy, dpy = dispersion
    delta = eta_to_delta(eta, beta0, xp)
    derivative = (1 + beta0 * beta0 * eta) / (1 + delta)
    if inverse:
        x, px, y, py = x - dx * delta, px - dpx * delta, y - dy * delta, py - dpy * delta
        z = z - derivative * (dx * px - dpx * x + dy * py - dpy * y)
    else:
        z = z + derivative * (dx * px - dpx * x + dy * py - dpy * y)
        x, px, y, py = x + dx * delta, px + dpx * delta, y + dy * delta, py + dpy * delta
    return x, px, y, py, z, eta


def _equivalent_transport(state, optics, phases, beta0, dispersion=(0., 0., 0., 0.), shear=0., inverse=False, xp=np):
    """D_equiv o shear o R o D_IP^-1, inverted with the current energy."""
    ip_dispersion = tuple(optics[name] for name in ('dx', 'dpx', 'dy', 'dpy'))
    state = _dispersion_lift(state, dispersion if inverse else ip_dispersion, beta0, inverse=True, xp=xp)
    if inverse:
        state = (*state[:4], state[4] - shear * eta_to_delta(state[5], beta0, xp), state[5])
    result = list(state)
    for index, plane in enumerate(('x', 'y')):
        beta, alpha, phase = optics['beta_' + plane], optics['alpha_' + plane], phases[index]
        sine, cosine = float(np.sin(phase)), float(np.cos(phase))
        a, b, c, d = cosine + alpha * sine, beta * sine, (alpha * cosine - sine) / beta, cosine
        u, pu = state[2 * index:2 * index + 2]
        if inverse:
            result[2 * index], result[2 * index + 1] = d * u - b * pu, -c * u + a * pu
        else:
            result[2 * index], result[2 * index + 1] = a * u + b * pu, c * u + d * pu
    if not inverse:
        result[4] = result[4] + shear * eta_to_delta(result[5], beta0, xp)
    return _dispersion_lift(result, ip_dispersion if inverse else dispersion, beta0, xp=xp)


def _equivalent_map(state, kind, optics, phases, strengths, beta0, phase=0., kappa=0., dispersion=(0., 0., 0., 0.), shear=0., xp=np):
    """T^-1 K T with the transverse and longitudinal kicks of one potential."""
    x, px, y, py, z, eta = _equivalent_transport(state, optics, phases, beta0, dispersion, shear, xp=xp)
    sx, sy = strengths
    if kind == 'theory':
        ax, ay = sx / (4 * optics['beta_x']**2), sy / (4 * optics['beta_y']**2)
        px, py = px + 2 * ax * z * x, py + 2 * ay * z * y
        eta = eta + ax * x * x + ay * y * y
    else:
        angle = phase - kappa * z
        sine, cosine = xp.sin(angle), xp.cos(angle)
        if kind == 'crab':
            px, py = px + sx * sine, py + sy * sine
            eta = eta - kappa * (sx * x + sy * y) * cosine
        elif kind == 'rfq':
            px, py = px - sx * x * cosine, py - sy * y * cosine
            eta = eta - kappa * (sx * x * x + sy * y * y) * sine / 2
        else:
            raise ValueError(f'Unknown IP-equivalent model {kind!r}')
    return _equivalent_transport((x, px, y, py, z, eta), optics, phases, beta0, dispersion, shear, inverse=True, xp=xp)


class _IPEquivalent(Command):
    """Shared implementation used by CrabCavity and FloatWaister only."""

    def _initialize(self, beam_id, sim, parameters, kind, phases, strengths):
        self.beam_id, self.s, self.order = beam_id, parameters.s, parameters.order
        self.cmd_name, self.cmd_type, self.length = parameters.name, self.__class__.__name__, 0.
        self.side, self.kind, self.is_enabled = parameters.side, kind, True
        self.optics_reference = parameters.optics_reference
        self.optics = resolve_ip_optics(sim, beam_id, self.optics_reference, expected_s=self.s)
        self.phases = tuple(float(np.arctan2(np.sin(phase), np.cos(phase))) for phase in phases)
        self.strengths = tuple(strengths)
        self.dispersion = tuple(getattr(parameters.equivalent_dispersion, key) for key in ('dx', 'dpx', 'dy', 'dpy'))
        self.shear, self._gpu_cache = parameters.longitudinal_shear, {}
        self.phase, self.frequency = 0., None
        if kind != 'theory':
            phase = parameters.phase or 0.
            # Reduce before adding clock phase or converting to particle precision.
            # sin/cos retain libm argument reduction for very large finite phases.
            self.phase = float(np.arctan2(np.sin(phase), np.cos(phase)))
            epoch = parameters.phase_epoch
            if epoch is None:
                epoch = sim.beams[beam_id].reference_program.origin
            self.frequency = LinearProgram(parameters.frequency, origin=epoch)

    def print(self):
        logging.getLogger(__name__).info('S=%g, Command=%s, Name=%s, Mode=%s, Side=%s, Strengths=%s', self.s, self.cmd_type, self.cmd_name, self.kind,
                                         self.side, self.strengths)

    def execute_cpu(self, sim):
        return self._execute(sim)

    def execute_gpu(self, sim):
        return self._execute(sim)

    def _execute(self, sim):
        beam = sim.beams[self.beam_id]
        for bunch in beam.bunches:
            if getattr(bunch, 'collision_frame', None) is not None:
                raise ValueError(f'{self.cmd_type} requires the ordinary IP frame')
            if not 0 < bunch.beta <= 1:
                raise ValueError('IP-equivalent elements require 0 < reference beta <= 1')
            phase, kappa = 0., 0.
            if self.frequency is not None:
                phase = float(self.phase + 2 * np.pi * self.frequency.phase_cycles(bunch.t0))
                kappa = float(2 * np.pi * self.frequency.values[0] / (bunch.beta * const.c))
            if beam.particles.xp is np:
                self._track_cpu(beam.particles, bunch, phase, kappa)
            else:
                self._track_gpu(beam.particles, bunch, phase, kappa)
        return True

    def _track_cpu(self, p, bunch, phase, kappa):
        beta = float(bunch.beta)
        for start in range(bunch.start_idx, bunch.end_idx, 16384):
            end = min(start + 16384, bunch.end_idx)
            active = np.flatnonzero(p.tag[start:end] > 0) + start
            state = tuple(getattr(p, name)[active] for name in ('x', 'px', 'y', 'py', 'z', 'dp'))
            if not np.all((state[5] > -1) & ((1 + state[5])**2 > state[1]**2 + state[3]**2)):
                raise ValueError('IP-equivalent elements require positive forward momentum')
            state = (*state[:5], delta_to_eta(state[5], beta))
            with np.errstate(invalid='ignore', divide='ignore'):
                result = _equivalent_map(state, self.kind, self.optics, self.phases, self.strengths, beta, phase, kappa, self.dispersion, self.shear)
                eta = result[5]
                delta = eta_to_delta(eta, beta)
            valid = (1 + beta**2 * eta > 0) & (1 + delta > 0) & ((1 + delta)**2 > result[1]**2 + result[3]**2)
            result = (*result[:5], delta)
            if not np.all(valid) or not all(np.all(np.isfinite(value)) for value in result):
                raise ValueError('IP-equivalent kick produced illegal particle coordinates')
            for name, value in zip(('x', 'px', 'y', 'py', 'z', 'dp'), result):
                getattr(p, name)[active] = value

    def _track_gpu(self, p, bunch, phase, kappa):
        import cupy as cp

        if bunch.start_idx == bunch.end_idx:
            return
        if 'kernel' not in self._gpu_cache:
            self._gpu_cache['kernel'] = _build_equivalent_kernel(cp)
        key = ('parameters', p.dtype.str)
        if key not in self._gpu_cache:
            values = [self.optics[name] for name in ('beta_x', 'alpha_x', 'beta_y', 'alpha_y', 'dx', 'dpx', 'dy', 'dpy')]
            values += list(self.dispersion) + list(self.phases) + list(self.strengths) + [self.shear]
            self._gpu_cache[key] = cp.asarray(values, dtype=p.dtype)
        selection = slice(bunch.start_idx, bunch.end_idx)
        status_key = ('invalid', bunch.end_idx - bunch.start_idx)
        if status_key not in self._gpu_cache:
            self._gpu_cache[status_key] = cp.empty(bunch.end_idx - bunch.start_idx, dtype=cp.int32)
        invalid = self._gpu_cache[status_key]
        kind, real = {'crab': 0, 'rfq': 1, 'theory': 2}[self.kind], p.dtype.type
        self._gpu_cache['kernel'](p.tag[selection], self._gpu_cache[key], real(bunch.beta), real((1 - bunch.beta) * (1 + bunch.beta)), real(phase),
                                  real(kappa), kind, *(getattr(p, name)[selection] for name in ('x', 'px', 'y', 'py', 'z', 'dp')), invalid)
        if bool(cp.any(invalid)):
            raise ValueError('IP-equivalent kick encountered illegal particle coordinates')


def _build_equivalent_kernel(cp):
    return cp.ElementwiseKernel(
        'int32 tag, raw T parameters, T beta, T rest_fraction, T phase, T kappa, int32 kind',
        'T x, T px, T y, T py, T z, T longitudinal, int32 invalid', r'''
invalid = 0;
if (tag > 0) {
    T q[6] = {x, px, y, py, z, longitudinal};
    bool valid = true;
    for (int j = 0; j < 6; ++j) {
        valid = valid && isfinite(q[j]);
    }
    valid = valid && q[5] > -1 && (1 + q[5]) * (1 + q[5]) > q[1] * q[1] + q[3] * q[3];
    T value = q[5] * (2 + q[5]);
    T momentum_ratio = 1 + q[5];
    q[5] = value / (sqrt(rest_fraction + beta * beta * momentum_ratio * momentum_ratio) + 1);
    T m[8];
    for (int u = 0; u < 2; ++u) {
        T b = parameters[2 * u], a = parameters[2 * u + 1];
        T sn = sin(parameters[12 + u]), cs = cos(parameters[12 + u]);
        m[4 * u] = cs + a * sn;
        m[4 * u + 1] = b * sn;
        m[4 * u + 2] = (a * cs - sn) / b;
        m[4 * u + 3] = cs;
    }
    for (int step = 0; step < 2; ++step) {
        int entry = step == 0 ? 4 : 8;
        value = 2 * q[5] + beta * beta * q[5] * q[5];
        T squared_ratio = (1 + 2 * q[5]) + beta * beta * q[5] * q[5];
        if (q[5] < 0 && beta > T(0.5)) {
            squared_ratio = (1 + q[5]) * (1 + q[5]) - rest_fraction * q[5] * q[5];
        }
        T delta = value / (sqrt(squared_ratio) + 1);
        T derivative = (1 + beta * beta * q[5]) / (1 + delta);
        T companion = 0;
        for (int j = 0; j < 4; ++j) {
            q[j] -= parameters[entry + j] * delta;
        }
        for (int u = 0; u < 2; ++u) {
            companion += parameters[entry + 2 * u] * q[2 * u + 1] - parameters[entry + 2 * u + 1] * q[2 * u];
        }
        q[4] -= derivative * companion;
        if (step == 1) {
            q[4] -= parameters[16] * delta;
        }
        for (int u = 0; u < 2; ++u) {
            T position = q[2 * u], momentum = q[2 * u + 1];
            if (step == 0) {
                q[2 * u] = m[4 * u] * position + m[4 * u + 1] * momentum;
                q[2 * u + 1] = m[4 * u + 2] * position + m[4 * u + 3] * momentum;
            } else {
                q[2 * u] = m[4 * u + 3] * position - m[4 * u + 1] * momentum;
                q[2 * u + 1] = -m[4 * u + 2] * position + m[4 * u] * momentum;
            }
        }
        if (step == 0) {
            q[4] += parameters[16] * delta;
        }
        int exit = step == 0 ? 8 : 4;
        companion = 0;
        for (int u = 0; u < 2; ++u) {
            companion += parameters[exit + 2 * u] * q[2 * u + 1] - parameters[exit + 2 * u + 1] * q[2 * u];
        }
        q[4] += derivative * companion;
        for (int j = 0; j < 4; ++j) {
            q[j] += parameters[exit + j] * delta;
        }
        if (step == 0) {
            T sx = parameters[14], sy = parameters[15], angle = phase - kappa * q[4];
            if (kind == 0) {
                q[1] += sx * sin(angle);
                q[3] += sy * sin(angle);
                q[5] -= kappa * (sx * q[0] + sy * q[2]) * cos(angle);
            } else if (kind == 1) {
                q[1] -= sx * q[0] * cos(angle);
                q[3] -= sy * q[2] * cos(angle);
                q[5] -= kappa * (sx * q[0] * q[0] + sy * q[2] * q[2]) * sin(angle) / 2;
            } else {
                T ax = sx / (4 * parameters[0] * parameters[0]), ay = sy / (4 * parameters[2] * parameters[2]);
                q[1] += 2 * ax * q[4] * q[0];
                q[3] += 2 * ay * q[4] * q[2];
                q[5] += ax * q[0] * q[0] + ay * q[2] * q[2];
            }
        }
    }
    value = 2 * q[5] + beta * beta * q[5] * q[5];
    T squared_ratio = (1 + 2 * q[5]) + beta * beta * q[5] * q[5];
    if (q[5] < 0 && beta > T(0.5)) {
        squared_ratio = (1 + q[5]) * (1 + q[5]) - rest_fraction * q[5] * q[5];
    }
    T delta = value / (sqrt(squared_ratio) + 1);
    valid = valid && 1 + beta * beta * q[5] > 0 && 1 + delta > 0 && (1 + delta) * (1 + delta) > q[1] * q[1] + q[3] * q[3];
    q[5] = delta;
    for (int j = 0; j < 6; ++j) {
        valid = valid && isfinite(q[j]);
    }
    if (valid) {
        x = q[0];
        px = q[1];
        y = q[2];
        py = q[3];
        z = q[4];
        longitudinal = q[5];
    } else {
        invalid = 1;
    }
}
''', 'pass_ip_equivalent')


@Command.register('crabcavity')
class CrabCavity(_IPEquivalent):

    def __init__(self, beam_id, sim, **command_kwargs):
        parameters = CrabCavityItem.model_validate(command_kwargs)
        phases, strengths = [0., 0.], [0., 0.]
        index = 0 if parameters.plane == 'x' else 1
        phases[index], strengths[index] = parameters.phase_advance, parameters.equivalent_kick
        self._initialize(beam_id, sim, parameters, 'crab', phases, strengths)
