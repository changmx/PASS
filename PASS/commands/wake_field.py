"""Thin integrated wake command; one independent physical state per instance."""
from functools import lru_cache
import logging
import hashlib

import numpy as np

from PASS.commands.command import Command
from PASS.para.schema.wake_field import WakeFieldItem as WakeFieldParameters
from .wake.wake_components import WakeComponent, SpatialTerm
from .wake.wake_models import ConstantWakeModel, ResonatorWakeModel, ResistiveWallWakeModel, TabulatedWakeModel
from .wake.wake_spectrum import ImpedanceSpectrum, SpectrumWakeModel, RationalWakeModel, fit_spectrum
from .wake.wake_velocity import VelocityLaw
from .wake.wake_wall import RoundWallImpedance
from .wake.wake_moments import WakeSourceProjector
from .wake.wake_state import WakeState
from .wake.wake_conventions import apply_kick_cpu

logger = logging.getLogger(__name__)


def _build_model(wake_config, longitudinal):
    values = wake_config.model_dump(exclude={"kind"})
    if wake_config.kind == "resistive_wall":
        beta, frequencies = values.pop("beta"), values.pop("frequencies")
        wall = RoundWallImpedance(**values)
        model = SpectrumWakeModel(wall.spectrum(frequencies, beta, longitudinal), "two_sided")
        model.wall_validity = wall.validity(frequencies, beta)
        return model
    if wake_config.kind in {"impedance", "fitted_impedance"}:
        spectrum = ImpedanceSpectrum(values.pop("frequencies"), values.pop("real"), values.pop("imag"), longitudinal)
        reconstruction = values.pop("reconstruction")
        if wake_config.kind == "impedance":
            return SpectrumWakeModel(spectrum, reconstruction)
        initial = [complex(*p) for p in values.pop("initial_poles")]
        if reconstruction == "causal_projection" and any(p.real >= 0 for p in initial):
            raise ValueError("Causal fit requires left-half-plane initial poles")
        return fit_spectrum(spectrum, initial, **values)
    if wake_config.kind == "modes":
        return RationalWakeModel([complex(*p) for p in values["poles"]], [complex(*r) for r in values["residues"]], longitudinal)
    return {
        "constant": ConstantWakeModel,
        "resonator": ResonatorWakeModel,
        "ultrarelativistic_wall": ResistiveWallWakeModel,
        "tabulated": TabulatedWakeModel
    }[wake_config.kind](**values)


def _build_component(config):
    velocity = (VelocityLaw("factorized", betas=(0., 1.), source=(1., 1.), witness=(1., 1.)) if config.velocity.kind == "ideal" else VelocityLaw(
        **config.velocity.model_dump()))
    spatial = None if config.spatial is None else SpatialTerm(**config.spatial.model_dump())
    longitudinal = config.component == "longitudinal" if spatial is None else spatial.plane == "z"
    if config.model.kind == "file":
        from .wake.wake_io import WakeConvention, read_wake_file
        values = config.model.model_dump(exclude={"kind", "convention", "file_path"})
        model = read_wake_file(config.model.file_path,
                               convention=WakeConvention(**config.model.convention.model_dump()),
                               component=config.component,
                               spatial=spatial,
                               **values)
    else:
        model = _build_model(config.model, longitudinal)
    return WakeComponent(config.component, model, config.scale, velocity, spatial)


@Command.register("wakefield")
class WakeField(Command):

    def __init__(self, beam_id, sim, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}
        self.cmd_name = kwargs.pop("name")
        kwargs.pop("command", None)
        self.configuration = WakeFieldParameters.model_validate(kwargs)
        wake_cfg = self.configuration
        if wake_cfg.groups is None:
            raise ValueError("Named wake configurations must be resolved by Config.load_input before construction")
        self.beam_id, self.s, self.length = beam_id, wake_cfg.s, 0.0
        self.cmd_type, self.is_enabled = "WakeField", wake_cfg.is_enabled
        self.slice_set_name = wake_cfg.slice_set
        self.component_groups = [tuple(_build_component(c) for c in group.components) for group in wake_cfg.groups]
        self.components = tuple(c for group in self.component_groups for c in group)
        for group, components in zip(wake_cfg.groups, self.component_groups):
            if group.boundary == "causal_passages" and any(not c.model.causal for c in components):
                raise ValueError("A two-sided response cannot use causal passage scheduling")
            if group.history == "state" and any(not c.model.causal for c in components):
                raise ValueError("Forward state history requires causal modes")
        self.projector = WakeSourceProjector()
        self.group_states = [WakeState() for _ in wake_cfg.groups]
        self._execution_plans = {}
        self.last_coefficients = None
        self.last_sources = None
        self.last_diagnostics = None

    @property
    def wake_state(self):
        """Single-group inspection; composite state is always group_states."""
        if len(self.group_states) != 1:
            raise ValueError("This location has multiple groups; inspect group_states")
        return self.group_states[0]

    def reset_state(self):
        for state in self.group_states:
            state.reset()
        self.last_coefficients = self.last_sources = None
        self.last_diagnostics = None

    def state_dict(self):
        identity = self._configuration_identity()
        return {
            "format": "PASS-wake-2",
            "configuration_sha256": identity,
            "groups": [state.state_dict() for state in self.group_states],
            "input_sha256": [getattr(c.model, "input_metadata", {}).get("sha256") for c in self.components]
        }

    def load_state_dict(self, data):
        expected = self._configuration_identity()
        if data.get("format") != "PASS-wake-2" or data.get("configuration_sha256") != expected:
            raise ValueError("Wake checkpoint does not match this physical model/configuration")
        if len(data["groups"]) != len(self.group_states):
            raise ValueError("Wake checkpoint group count does not match")
        file_hashes = [getattr(c.model, "input_metadata", {}).get("sha256") for c in self.components]
        if data.get("input_sha256", [None] * len(self.components)) != file_hashes:
            raise ValueError("Wake checkpoint input file contents do not match")
        candidates = [WakeState.from_state_dict(d) for d in data["groups"]]
        for config, components, state in zip(self.configuration.groups, self.component_groups, candidates):
            if state.mode_amplitudes and config.history != "state":
                raise ValueError("Mode state is inconsistent with the selected history algorithm")
            if state.history and config.history != "direct":
                raise ValueError("Source history is inconsistent with the selected algorithm")
            if state.convolution is not None and config.history != "partitioned":
                raise ValueError("Convolution state is inconsistent with the selected algorithm")
            if config.history == "partitioned" and state.last_turn is not None and state.convolution is None:
                raise ValueError("Checkpoint is missing convolution history")
            for ci, vector in state.mode_amplitudes.items():
                if not isinstance(ci, int) or not 0 <= ci < len(components):
                    raise ValueError("Invalid mode component index in checkpoint")
                size = 2 if config.solver == "recursive" else len(components[ci].model.poles)
                if len(vector) != size:
                    raise ValueError("Checkpoint mode dimension does not match the model")
        self.group_states = candidates
        self.last_coefficients = self.last_sources = self.last_diagnostics = None

    def _configuration_identity(self):
        # Adding an optional time-grid field must not invalidate checkpoints
        # made with the existing fixed-grid/direct/modal configurations.
        exclude = {"configuration": True}
        if all(g.time_grid is None for g in self.configuration.groups):
            exclude["groups"] = {"__all__": {"time_grid"}}
        return hashlib.sha256(self.configuration.model_dump_json(exclude=exclude).encode()).hexdigest()

    def execute_cpu(self, sim):
        return self._execute(sim, "cpu")

    def execute_gpu(self, sim):
        return self._execute(sim, "gpu")

    def _execute(self, sim, backend):
        if not self.is_enabled:
            return False
        from .wake.execution import GroupExecution
        gpu = backend == "gpu"
        cfg, beam, turn = self.configuration, sim.beams[self.beam_id], sim.state.turn
        if gpu and isinstance(beam.particles.z, np.ndarray):
            raise TypeError("GPU wake tracking requires device particle arrays")
        if gpu:
            import cupy as xp
        else:
            xp = np
        if any(b.slice_sets.get(cfg.slice_set) is not None and b.slice_sets[cfg.slice_set].periodic for b in beam.bunches):
            if any(b.slice_sets[cfg.slice_set].valid_s != self.s for b in beam.bunches):
                raise ValueError("Periodic wake requires Slicer at the same location")
            if any(g.boundary != "causal_passages" for g in cfg.groups):
                raise ValueError("Evolving periodic arrival slices require causal_passages wake history")
        plans = self._execution_plans.get(backend)
        if plans is None:
            plans = [GroupExecution(g, c, backend) for g, c in zip(cfg.groups, self.component_groups)]
            self._execution_plans[backend] = plans
        project = self.projector.project_gpu if gpu else self.projector.project_cpu
        projection = project(beam, cfg.slice_set, turn, self.components, "uniform")
        results, candidates, updates, diagnostics = [], [], [], []
        for plan, state in zip(plans, self.group_states):
            value, candidate, update, diagnostic = plan.preview(projection.sources, state, turn)
            results.append(value)
            candidates.append(candidate)
            updates.append(update)
            diagnostics.append(diagnostic)
        coefficients = results[0] if len(results) == 1 else xp.concatenate(results, axis=0)
        from .wake.wake_state import finite_gpu
        if not (finite_gpu(self, coefficients) if gpu else bool(np.all(np.isfinite(coefficients)))):
            raise FloatingPointError("Wake solver returned a non-finite voltage; particle coordinates were not updated")
        if gpu:
            kick_gpu(self, beam, projection, coefficients, turn)
        else:
            p = beam.particles
            for bunch, indices, ids in projection.witnesses:
                voltage = np.zeros((3, len(indices)))
                for ci, component in enumerate(self.components):
                    a, b = component.test_powers
                    plane = {"z": 0, "x": 1, "y": 2}[component.plane]
                    voltage[plane] += coefficients[ci, ids] * p.x[indices]**a * p.y[indices]**b
                apply_kick_cpu(p, bunch, indices, voltage, s=self.s, turn=turn)
        for update in updates:
            if update is not None:
                update.commit()
        self.group_states = candidates
        self.last_coefficients, self.last_sources = coefficients, projection.sources
        self.last_diagnostics = diagnostics
        return True

    def print(self):
        logger.info("S=%.4f, Command=WakeField, Name=%s, Slice set=%s, Groups=%s, Components=%d", self.s, self.cmd_name, self.slice_set_name,
                    [(g.name, g.solver, g.history) for g in self.configuration.groups], len(self.components))


# GPU: fused device kicks

_GPU_CODE = r'''
#if FLOAT_PARTICLES
using R = float;
#else
using R = double;
#endif
__device__ void mechanical(
    int i,
    R* px,
    R* py,
    R* dp,
    int* tag,
    int* lost_turn,
    float* lost_position,
    const double* v,
    double p0,
    double m0,
    double za,
    double position,
    int turn
) {
    double oldp = (1 + (double)dp[i]) * p0, olde = hypot(oldp, m0), de = -za * v[0], energy = olde + de;
    double inverse = oldp > 0 ? (za / p0) * (olde / oldp) : 0.;
    double dx = (double)px[i] + v[1] * inverse, dy = (double)py[i] + v[2] * inverse;
    double delta = (double)dp[i];
    if (v[0] != 0.) {
        double p = sqrt(fmax(0., (energy - m0) * (energy + m0))), denominator = p0 * (p + oldp);
        if (denominator != 0.)
            delta += de * (2 * olde + de) / denominator;
    }
    if (oldp > 0 && energy > m0 && isfinite(delta) && isfinite(dx) && isfinite(dy) && (1 + delta) * (1 + delta) > dx * dx + dy * dy) {
        if (v[0] != 0.)
            dp[i] = (R)delta;
        px[i] = (R)dx;
        py[i] = (R)dy;
    } else {
        tag[i] = -abs(tag[i]);
        lost_turn[i] = turn;
        lost_position[i] = (float)position;
    }
}
__device__ double monomial(
    double x,
    int n
) {
    double result = 1.;
    while (n) {
        if (n & 1)
            result *= x;
        n >>= 1;
        if (n)
            x *= x;
    }
    return result;
}
extern "C" __global__ void kick(
    R* px,
    R* py,
    R* dp,
    const R* x,
    const R* y,
    int* tag,
    int* lost_turn,
    float* lost_position,
    const int* ids,
    const double* coeff,
    int start,
    int end,
    int n_slices,
    int offset,
    int total,
    int nc,
    const int* planes,
    const int* powers,
    double p0,
    double m0,
    double za,
    double position,
    int turn
) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end || tag[i] <= 0)
        return;
    int id = ids[i - start];
    if (id < 0 || id >= n_slices)
        return;
    double v[3] = {0., 0., 0.};
    for (int k = 0; k < nc; k++) {
        double value = coeff[k * total + offset + id];
        value *= monomial((double)x[i], powers[2 * k]) * monomial((double)y[i], powers[2 * k + 1]);
        v[planes[k]] += value;
    }
    mechanical(i, px, py, dp, tag, lost_turn, lost_position, v, p0, m0, za, position, turn);
}
extern "C" __global__ void kick_batch(
    R* px,
    R* py,
    R* dp,
    const R* x,
    const R* y,
    int* tag,
    int* lost_turn,
    float* lost_position,
    const int* layout,
    const unsigned long long* ptr,
    const double* reference,
    const double* coeff,
    int total,
    int nc,
    const int* planes,
    const int* powers,
    double position,
    int turn
) {
    int b = blockIdx.y, start = layout[4 * b], end = layout[4 * b + 1], n_slices = layout[4 * b + 2], offset = layout[4 * b + 3];
    const int* ids = (const int*)ptr[3 * b];
    for (int i = start + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += blockDim.x * gridDim.x) {
        if (tag[i] <= 0)
            continue;
        int id = ids[i - start];
        if (id < 0 || id >= n_slices)
            continue;
        double v[3] = {0., 0., 0.};
        for (int k = 0; k < nc; k++)
            v[planes[k]] += coeff[k * total + offset + id] * monomial((double)x[i], powers[2 * k]) * monomial((double)y[i], powers[2 * k + 1]);
        mechanical(i, px, py, dp, tag, lost_turn, lost_position, v, reference[3 * b], reference[3 * b + 1], reference[3 * b + 2], position, turn);
    }
}
'''


@lru_cache(maxsize=None)
def _get_gpu_kernels(dtype, device):
    import cupy as cp

    with cp.cuda.Device(device):
        names = ('kick', 'kick_batch')
        module = cp.RawModule(code=_GPU_CODE,
                              options=("--std=c++17", f"-DFLOAT_PARTICLES={int(np.dtype(dtype) == np.float32)}"),
                              name_expressions=names)
        return {name: module.get_function(name) for name in names}


def kick_gpu(command, beam, projection, coefficients, turn):
    import cupy as cp
    from .wake.wake_models import device_arrays
    from .wake.wake_conventions import signed_charge_per_mass_unit
    planes, powers = device_arrays(command, "kick_layout", (np.array([{
        "z": 0,
        "x": 1,
        "y": 2
    }[c.plane] for c in command.components], dtype=np.int32), np.array([c.test_powers for c in command.components], dtype=np.int32)))
    p = beam.particles
    if hasattr(projection, "device_layout"):
        layout, pointers, blocks, _ = projection.device_layout
        reference = cp.asarray([(bunch.p0, bunch.m0, signed_charge_per_mass_unit(bunch)) for bunch, *_ in projection.witnesses], dtype=cp.float64)
        _get_gpu_kernels(p.dtype.str, cp.cuda.runtime.getDevice())["kick_batch"](
            (blocks, len(projection.witnesses)), (256, ),
            (p.px, p.py, p.dp, p.x, p.y, p.tag, p.lost_turn, p.lost_position, layout, pointers, reference, coefficients,
             np.int32(coefficients.shape[1]), np.int32(len(command.components)), planes, powers, np.float64(command.s), np.int32(turn)))
        return
    for bunch, ids, offset, n_slices in projection.witnesses:
        start, end = bunch.start_idx, bunch.end_idx
        if start == end:
            continue
        _get_gpu_kernels(p.dtype.str, cp.cuda.runtime.getDevice())["kick"](
            ((end - start + 255) // 256, ), (256, ),
            (p.px, p.py, p.dp, p.x, p.y, p.tag, p.lost_turn, p.lost_position, ids, coefficients, np.int32(start), np.int32(end), np.int32(n_slices),
             np.int32(offset), np.int32(coefficients.shape[1]), np.int32(len(command.components)), planes, powers, np.float64(
                 bunch.p0), np.float64(bunch.m0), np.float64(signed_charge_per_mass_unit(bunch)), np.float64(command.s), np.int32(turn)))
