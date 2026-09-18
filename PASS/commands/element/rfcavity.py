"""Physical, shared RF waveforms and exact zero-length energy kicks.

z = beta0*c*(t0-t). CPU arrays and a fused CUDA kernel implement the same
map with FP64 intermediates; particle storage keeps its configured precision.
"""
import logging
from typing import NamedTuple

import numpy as np

from PASS.commands.command import Command
from PASS.utils.program import LinearProgram
from PASS.core.bunch import set_reference_energy
from PASS.para.schema.rf import RFComponent
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu, check_aperture_gpu

logger = logging.getLogger(__name__)


def _component_parameters(raw):
    if isinstance(raw, RFComponent):
        return raw
    aliases = {field.alias.lower(): name for name, field in RFComponent.model_fields.items()}
    values = {aliases.get(k.lower(), k): v for k, v in raw.items()}
    return RFComponent.model_validate(values)


class _ReferenceKick(NamedTuple):
    """Host reference scalars shared by the CPU and CUDA maps."""
    charge: float
    energy: float
    momentum: float
    scale: float
    delta: float
    z_scale: float


class RFWaveform:
    """One physical voltage component, shared by every bunch in this command."""

    def __init__(self, raw, reference):
        parameters = _component_parameters(raw)
        if parameters.program_file:
            import tfs
            table = tfs.read(parameters.program_file)
            table.columns = table.columns.str.lower()
            required = {'time', 'voltage', 'phase'}
            if parameters.harmonic is None:
                required.add('frequency')
            if not required.issubset(table.columns):
                raise ValueError(f'RF program requires columns {sorted(required)}')
            if parameters.harmonic is not None and 'frequency' in table.columns:
                raise ValueError('Harmonic RF file must not also define FREQUENCY')
            parameters = RFComponent(times=table.time.tolist(),
                                     voltage=table.voltage.tolist(),
                                     phase=table.phase.tolist(),
                                     harmonic=parameters.harmonic,
                                     frequency=table.frequency.tolist() if parameters.harmonic is None else None)
        self.harmonic = parameters.harmonic or 1
        self.frequency = reference if parameters.harmonic is not None else LinearProgram(
            parameters.frequency, parameters.times, origin=reference.origin)
        self.voltage = LinearProgram(parameters.voltage, parameters.times, origin=reference.origin)
        self.phase = LinearProgram(parameters.phase, parameters.times, origin=reference.origin)

    def value(self, reference_time, offset=0., xp=np):
        cycles = self.frequency.phase_cycles(reference_time, offset, xp)
        phase = (self.phase.values[0] if len(self.phase.values) == 1 else self.phase.value(reference_time, offset, xp))
        angle = 2 * np.pi * xp.remainder(self.harmonic * cycles, 1.) + phase
        voltage = (self.voltage.values[0] if len(self.voltage.values) == 1 else self.voltage.value(reference_time, offset, xp))
        return voltage * xp.sin(angle)


@Command.register('rfcavity')
class RFCavity(Command):
    """Simultaneous effective-voltage components at one physical location.

    Inputs: Components, S (m), Is enabled, Dp aperture, transverse aperture.
    Scalar legacy RF inputs and turn-row RF tables are deliberately rejected.
    Frequencies are prescribed physical functions, never inferred from the
    instantaneous energy of a tracked bunch. Positive z means earlier arrival.
    """

    def __init__(self, beam_id, sim, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}
        obsolete = {'voltage (v)', 'harmonic', 'phase (rad)', 'phi offset (rad)', 'rf data file'} & kwargs.keys()
        if obsolete:
            raise ValueError(f'RFCavity requires Components; removed scalar/turn inputs: {sorted(obsolete)}')
        if float(kwargs.get('length (m)', 0.)) != 0.:
            raise ValueError('RFCavity is a zero-length effective-voltage kick')
        self.beam_id, self.s = beam_id, float(kwargs['s (m)'])
        self.cmd_name, self.cmd_type, self.length = kwargs['name'], 'RFCavity', 0.
        self.is_enabled = kwargs.get('is enabled', True)
        if not isinstance(self.is_enabled, bool):
            raise ValueError('Is enabled must be a boolean')
        self.aperture_type = kwargs.get('aperture type', 'off').lower()
        self.aperture_value = kwargs.get('aperture value', [])
        limits = kwargs.get('dp aperture')
        self.dp_aperture_lower, self.dp_aperture_upper = (-np.inf, np.inf) if limits is None else limits
        if (not np.isfinite(self.s) or (limits is not None and not np.all(np.isfinite(limits)))
                or not self.dp_aperture_lower < self.dp_aperture_upper):
            raise ValueError('RF position and ordered acceptance bounds must be finite')
        raw = kwargs.get('components')
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError('RFCavity requires a nonempty Components list')
        beam = sim.beams[beam_id]
        # Explicit-frequency components need only a common time origin. A
        # harmonic component requires the prescribed machine reference program.
        reference = getattr(beam, 'reference_program', None)
        if reference is None:
            if any(_component_parameters(v).harmonic is not None for v in raw):
                raise ValueError('Harmonic RF requires beam.reference_program')
            reference = LinearProgram(1., origin=0.)
        self.components = tuple(RFWaveform(v, reference) for v in raw)
        self._cuda = {}
        super().__init__()

    def print(self):
        logger.info('S=%g, Command=RFCavity, Name=%s, Components=%d, Enabled=%s', self.s, self.cmd_name, len(self.components), self.is_enabled)

    def execute_cpu(self, sim):
        return self._execute(sim)

    def execute_gpu(self, sim):
        return self._execute(sim)

    def _execute(self, sim):
        if not self.is_enabled:
            return False
        beam, turn = sim.beams[self.beam_id], sim.state.turn
        xp = beam.particles.xp
        aperture = check_aperture_cpu if xp is np else check_aperture_gpu
        for bunch in beam.bunches:
            self._track(beam, bunch, turn)
            aperture(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)
        return True

    def _track(self, beam, bunch, turn):
        if beam.particles.xp is not np:
            return self._track_gpu(beam, bunch, turn)
        p, xp = beam.particles, beam.particles.xp
        bunch_slice = slice(bunch.start_idx, bunch.end_idx)
        px, py, z, dp, tag = (getattr(p, k)[bunch_slice] for k in ('px', 'py', 'z', 'dp', 'tag'))
        alive = tag > 0
        beta_old, p0_old = bunch.beta, bunch.p0
        ref = self._reference_kick(bunch)
        # All components sample the same entry event, before any coordinate or
        # energy is changed. No full particle clock array is stored.
        offset = -z.astype(xp.float64) / (beta_old * const.c)
        gain = xp.zeros(z.shape, dtype=xp.float64)
        for component in self.components:
            gain += ref.charge * component.value(bunch.t0, offset, xp)
        delta = dp.astype(xp.float64)
        old_p = p0_old * (1. + delta)
        old_e = xp.hypot(old_p, bunch.m0)
        new_e = old_e + gain
        momentum_sq = (new_e - bunch.m0) * (new_e + bunch.m0)
        transverse_sq = p0_old**2 * (px.astype(xp.float64)**2 + py.astype(xp.float64)**2)
        forward = xp.isfinite(new_e) & (new_e > bunch.m0) & (old_p > 0) & (momentum_sq > transverse_sq)
        active = alive & forward
        stopped = alive & ~forward
        tag[stopped] = -xp.abs(tag[stopped])
        new_p = xp.sqrt(xp.where(active, momentum_sq, old_p**2))
        applied = xp.where(active, gain, 0.)
        # Stable weak-kick conversion. Do not subtract nearly equal momenta.
        kick_delta = applied * (2 * old_e + applied) / (ref.momentum * xp.where(new_p + old_p > 0, new_p + old_p, 1.))
        dp[:] = xp.where(active, delta * ref.scale + ref.delta + kick_delta, dp)
        px[:] = xp.where(active, px.astype(xp.float64) * ref.scale, px)
        py[:] = xp.where(active, py.astype(xp.float64) * ref.scale, py)
        z[:] = xp.where(active, z.astype(xp.float64) * ref.z_scale, z)
        set_reference_energy(bunch, ref.energy)
        # The physical reference arrival time and every saved slice interval
        # stay unchanged. A user Slicer command is the only rebinning operation.
        outside = active & ((dp < self.dp_aperture_lower) | (dp > self.dp_aperture_upper))
        tag[outside] = -xp.abs(tag[outside])
        lost = alive & (tag < 0)
        p.lost_position[bunch_slice][lost] = self.s
        p.lost_turn[bunch_slice][lost] = turn

    def _reference_kick(self, bunch):
        """Validate and compute the reference kick before changing any state."""
        beta_old, p0_old = bunch.beta, bunch.p0
        if not 0 < beta_old < 1:
            raise ValueError('RF requires a finite massive-particle reference velocity')
        charge = np.sign(bunch.num_charge) * bunch.qm_ratio
        reference_gain = sum(charge * float(c.value(bunch.t0)) for c in self.components)
        old_total = bunch.Ek + bunch.m0
        new_total = old_total + reference_gain
        if not np.isfinite(new_total) or new_total <= bunch.m0:
            raise ValueError(f'RFCavity {self.cmd_name}: reference total energy must exceed rest energy')
        p0_new = np.sqrt((new_total - bunch.m0) * (new_total + bunch.m0))
        scale = p0_old / p0_new
        reference_delta = -reference_gain * (2 * old_total + reference_gain) / (p0_new * (p0_new + p0_old))
        return _ReferenceKick(charge, new_total, p0_new, scale, reference_delta, (p0_new / new_total) / beta_old)

    def _track_gpu(self, beam, bunch, turn):
        import cupy as cp

        p = beam.particles
        ref = self._reference_kick(bunch)
        start, end = bunch.start_idx, bunch.end_idx
        if end > start:
            key = (cp.cuda.runtime.getDevice(), np.dtype(p.dtype))
            if key not in self._cuda:
                self._cuda[key] = _prepare_rf_kernel(self.components, p.dtype, cp)
            kernel, template, tables = self._cuda[key]
            # Pass the small waveform descriptors by value. No per-particle
            # scratch arrays or device-to-host synchronization are required.
            parameters = template.copy()
            for j, component in enumerate(self.components):
                base, value, index = component.frequency.phase_anchor(bunch.t0)
                parameters['components']['frequency']['base'][0, j] = base
                parameters['components']['frequency']['value'][0, j] = value
                parameters['components']['frequency']['padding'][0, j] = index
            if len(self.components) > 32:
                # Large component lists exceed the portable 4 KiB argument
                # bank. The same kernel reads one packed descriptor buffer.
                parameters = cp.asarray(parameters.view(np.float64))
            threads = 256
            kernel(((end - start + threads - 1) // threads, ), (threads, ),
                   (p.px, p.py, p.z, p.dp, p.tag, p.lost_position, p.lost_turn, np.int32(start), np.int32(end), np.int32(turn), parameters,
                    np.float64(bunch.t0), np.float64(1 / (bunch.beta * const.c)), np.float64(ref.charge), np.float64(bunch.m0), np.float64(
                        bunch.p0), np.float64(ref.momentum), np.float64(ref.scale), np.float64(ref.delta), np.float64(
                            ref.z_scale), np.float64(self.dp_aperture_lower), np.float64(self.dp_aperture_upper), np.float64(self.s)))
        set_reference_energy(bunch, ref.energy)


def _prepare_rf_kernel(components, dtype, cp):
    """Cache waveform tables on this device and compile the component count.

    Explicit 8-byte alignment matches the CUDA structs below. Tables are
    immutable prescribed inputs, retained with the command for pointer life.
    """
    program_dtype = np.dtype([('data', np.uint64), ('count', np.int32), ('padding', np.int32), ('base', np.float64), ('value', np.float64)],
                             align=True)
    component_dtype = np.dtype([('frequency', program_dtype), ('voltage', program_dtype), ('phase', program_dtype), ('harmonic', np.float64)],
                               align=True)
    parameters = np.zeros(1, dtype=np.dtype([('components', component_dtype, (len(components), ))], align=True))
    tables, stored = [], {}
    scalar = True
    for j, component in enumerate(components):
        row = parameters['components'][0, j]
        row['harmonic'] = component.harmonic
        for name in ('frequency', 'voltage', 'phase'):
            program = getattr(component, name)
            record = row[name]
            record['count'] = len(program.values)
            record['value'] = program.values[0]
            record['base'] = program.integral_origin
            if len(program.values) > 1:
                scalar = False
                if id(program) not in stored:
                    table = cp.asarray(np.concatenate((program.times, program.values, program.slopes, program.phase_integrals)))
                    stored[id(program)] = table
                    tables.append(table)
                record['data'] = stored[id(program)].data.ptr
    kernel = cp.RawKernel(_RF_CUDA,
                          'track_rf',
                          options=('--std=c++14', f'-DRF_FLOAT={int(np.dtype(dtype)==np.dtype(np.float32))}', f'-DRF_COMPONENTS={len(components)}',
                                   f'-DRF_SCALAR={int(scalar)}', f'-DRF_INDIRECT={int(len(components)>32)}'))
    return kernel, parameters, tables


_RF_CUDA = r'''
#if RF_FLOAT
using real_t = float;
#else
using real_t = double;
#endif
struct Program {
    const double* data;
    int count, padding;
    double base, value;
};
struct Component {
    Program frequency, voltage, phase;
    double harmonic;
};
struct Components {
    Component components[RF_COMPONENTS];
};
static_assert(
    sizeof(Program) == 32,
    "RF program ABI mismatch"
);
static_assert(
    sizeof(Component) == 104,
    "RF component ABI mismatch"
);

__device__ __forceinline__ double unit_cycle(
    double x
) {
    // Positive modulo, including negative arrival offsets.
    return x - floor(x);
}

__device__ __forceinline__ double program_value(
    const Program& p,
    double reference,
    double offset,
    bool integral
) {
    if (p.count == 1)
        return integral ? unit_cycle(p.base + p.value * offset) : p.value;
    const double* times = p.data;
    const double* values = times + p.count;
    const double* slopes = values + p.count;
    const double* integrals = slopes + p.count;
    int left = 0, right = p.count;
    // Compare local offsets: never form reference+offset, which can erase
    // intra-bunch times at large epochs. Use the right side at a knot.
    while (left < right) {
        int middle = left + (right - left) / 2;
        if (offset < times[middle] - reference)
            right = middle;
        else
            left = middle + 1;
    }
    int index = left > 0 ? left - 1 : 0;
    double dx = (reference - times[index]) + offset;
    double slope = (reference - times[0]) + offset < 0.0 ? 0.0 : slopes[index];
    if (integral) {
        bool before = offset < times[0] - reference;
        if (index == p.padding && before == (reference < times[0]))
            return unit_cycle(p.base + offset * (p.value + 0.5 * slope * offset));
        int anchor = index;
        if (!before && index + 1 < p.count && fabs(reference - times[index + 1]) < fabs(reference - times[index]))
            anchor = index + 1;
        double local_dx = (reference - times[anchor]) + offset;
        return unit_cycle(integrals[anchor] + local_dx * (values[anchor] + 0.5 * slope * local_dx));
    }
    return values[index] + slope * dx;
}

extern "C" __global__ void track_rf(
    real_t* __restrict__ px,
    real_t* __restrict__ py,
    real_t* __restrict__ z,
    real_t* __restrict__ dp,
    int* __restrict__ tag,
    float* __restrict__ lost_position,
    int* __restrict__ lost_turn,
    int start,
    int end,
    int turn,
#if RF_INDIRECT
    const Components* parameters_ptr,
#else
    const Components parameters,
#endif
    double reference,
    double inverse_velocity,
    double charge,
    double mass,
    double p0_old,
    double p0_new,
    double scale,
    double reference_delta,
    double z_scale,
    double lower,
    double upper,
    double position
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end || tag[i] <= 0)
        return;
#if RF_INDIRECT
    const Components& parameters = *parameters_ptr;
#endif
    const double offset = -(double)z[i] * inverse_velocity;
    double gain = 0.0;
#if RF_COMPONENTS <= 8
#pragma unroll
#endif
    for (int j = 0; j < RF_COMPONENTS; ++j) {
        const Component& c = parameters.components[j];
#if RF_SCALAR
        double cycles = unit_cycle(c.frequency.base + c.frequency.value * offset);
        double voltage = c.voltage.value, phase = c.phase.value;
#else
        double cycles = program_value(c.frequency, reference, offset, true);
        double voltage = program_value(c.voltage, reference, offset, false);
        double phase = program_value(c.phase, reference, offset, false);
#endif
        double angle = 6.283185307179586476925286766559 * unit_cycle(c.harmonic * cycles) + phase;
        // Round each component before summation, matching simultaneous CPU
        // accumulation instead of fusing the last multiply into the sum.
        gain = __dadd_rn(gain, charge * (voltage * sin(angle)));
    }
    double delta = (double)dp[i], old_p = p0_old * (1.0 + delta);
    // Direct squares are safe throughout the physical beam range. Retain
    // scaled hypot for extreme inputs that could overflow either square.
    double old_e = (fabs(old_p) < 1e150 && mass < 1e150) ? sqrt(old_p * old_p + mass * mass) : hypot(old_p, mass);
    double new_e = old_e + gain;
    double momentum_sq = (new_e - mass) * (new_e + mass);
    double px0 = (double)px[i], py0 = (double)py[i];
    double transverse_sq = p0_old * p0_old * (px0 * px0 + py0 * py0);
    bool forward = isfinite(new_e) && new_e > mass && old_p > 0.0 && momentum_sq > transverse_sq;
    if (forward) {
        double new_p = sqrt(momentum_sq);
        double kick_delta = gain * (2.0 * old_e + gain) / (p0_new * (new_p + old_p));
        dp[i] = (real_t)(delta * scale + reference_delta + kick_delta);
        px[i] = (real_t)(px0 * scale);
        py[i] = (real_t)(py0 * scale);
        z[i] = (real_t)((double)z[i] * z_scale);
        // Acceptance uses the stored final delta, exactly as on CPU.
        if (!((double)dp[i] < lower || (double)dp[i] > upper))
            return;
    }
    // Stopped particles retain their entry coordinates; acceptance losses
    // retain their post-kick coordinates. Earlier loss records are untouched.
    tag[i] = -tag[i];
    lost_position[i] = (float)position;
    lost_turn[i] = turn;
}
'''
