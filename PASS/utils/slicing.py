"""Shared body-slicing utilities and midpoint space-charge scheduling.

Machine positions use the element exit coordinate (the MAD-X S convention).
Yoshida's signed internal stages never determine the SC integration weight.
"""

from dataclasses import dataclass
from functools import lru_cache
import logging
import math
from numbers import Integral

import numpy as np

from PASS.utils.constants import const

logger = logging.getLogger(__name__)


def resolve_internal_sc_aperture(config, parent_type, parent_value, name, sim, beam_id):
    """Use the element aperture for both deposition walls and particle losses."""
    from PASS.para.schema.space_charge import validate_loss_aperture

    kind = str(parent_type).strip().lower()
    dimensions = validate_loss_aperture(kind, parent_value)
    if kind == "default":
        # Generic element default is +/-1 m, not the SC grid rectangle.
        kind, dimensions = "rectangle", [1.0, 1.0]
    elif kind == "off":
        dimensions = []
    if config.aperture_type != "default" and (config.aperture_type != kind or config.aperture_value != dimensions):
        key = (beam_id, name, config.aperture_type, repr(config.aperture_value), kind, repr(dimensions))
        warned = getattr(sim, "_internal_sc_aperture_warnings", None)
        if warned is None:
            warned = sim._internal_sc_aperture_warnings = set()
        if key not in warned:
            logger.warning("Element %r internal SC aperture %s %s differs from element aperture %s %s; "
                           "using the element aperture", name, config.aperture_type, config.aperture_value, kind, dimensions)
            warned.add(key)
    return config.model_copy(deep=True, update={"aperture_type": kind, "aperture_value": dimensions})


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class SCNode:
    index: int
    slice_index: int
    placement: str
    offset: float
    length: float


@dataclass(frozen=True)
class SlicePlan:
    length: float
    requested_slices: int
    num_slices: int
    slices_per_sc: int
    nodes: tuple[SCNode, ...]

    @property
    def slice_length(self):
        return self.length / self.num_slices


def make_slice_plan(length, num_slices=1, num_kicks=0):
    """Build a deterministic plan without particles or solver resources."""
    length = float(length)
    if not math.isfinite(length) or length < 0:
        raise ValueError("Element length must be finite and non-negative")
    num_slices = positive_integer(num_slices, "Num slices")
    if isinstance(num_kicks, bool) or not isinstance(num_kicks, Integral) or num_kicks < 0:
        raise ValueError("SC Num kicks must be a non-negative integer")
    if not num_kicks:
        return SlicePlan(length, num_slices, num_slices, 0, ())
    if length <= const.eps:
        raise ValueError("Internal Space charge requires a thick element with positive length")
    m = (num_slices + num_kicks - 1) // num_kicks
    sc_length = length / num_kicks
    placement = "center" if m % 2 else "boundary"
    nodes = tuple(SCNode(j, j * m + (m - 1) // 2, placement, (j + 0.5) * sc_length, sc_length) for j in range(num_kicks))
    return SlicePlan(length, num_slices, m * num_kicks, m, nodes)


def configure_element_slicing(element, sim, values):
    """Configure once; preserve the requested external slice count."""
    requested = positive_integer(getattr(element, "num_slice", values.get("num slices", 1)), "Num slices")
    element.num_slice = requested
    element.slice_plan = make_slice_plan(element.length, requested)
    element._sc_nodes = {}
    element._sc_sim = None
    raw = values.get("space charge")
    element._sc_requested = raw is not None
    if raw is None:
        return
    settings = getattr(sim.cfg, "space_charge", [])
    if element.beam_id >= len(settings) or not settings[element.beam_id].enabled:
        return
    from PASS.para.schema.space_charge import parse_element_space_charge

    config = parse_element_space_charge(raw)
    config = resolve_internal_sc_aperture(config, element.aperture_type, element.aperture_value, element.cmd_name, sim, element.beam_id)
    element.slice_plan = make_slice_plan(element.length, requested, config.num_kicks)
    if not math.isfinite(element.s):
        raise ValueError("Element S (m) must be finite")
    from PASS.commands.space_charge import SpaceCharge
    command_values = config.model_dump(by_alias=True)
    command_values.pop("Num kicks")
    for node in element.slice_plan.nodes:
        command = SpaceCharge(
            element.beam_id, sim, **command_values, **{
                "Name": element.cmd_name,
                "S (m)": element.s - element.length + node.offset,
                "SC start (m)": element.s - element.length + node.offset - node.length / 2,
                "SC length (m)": node.length
            })
        command.parent_element = element.cmd_name
        command.internal_node_index = node.index
        element._sc_nodes[node.slice_index] = (node, command)
    element._sc_sim = sim
    logger.info("%s: external slices requested=%d actual=%d, SC kicks=%d, placement=%s, SC length=%g m", element.cmd_name, requested,
                element.slice_plan.num_slices, config.num_kicks, element.slice_plan.nodes[0].placement, element.slice_plan.nodes[0].length)


def print_element_slicing(element):
    """Report the effective plan and SC aperture from element.print()."""
    plan = element.slice_plan
    errors = getattr(element, "field_errors", None)
    if errors is not None:
        logger.info("  Absolute field errors: enabled=%s, KNL=%s, KSL=%s", errors.enabled, errors.knl.tolist(), errors.ksl.tolist())
    logger.info("  Slicing: requested=%d, actual=%d, external slice length=%g m", plan.requested_slices, plan.num_slices, plan.slice_length)
    if not element._sc_nodes:
        status = "disabled by top-level Space charge.Enabled" if element._sc_requested else "off"
        logger.info("  Internal SC: %s", status)
        return
    commands = [command for _, command in element._sc_nodes.values()]
    first, last = commands[0], commands[-1]
    logger.info("  Internal SC: Configuration=%s, Method=%s, Solver=%s, Slice set=%s, "
                "Num kicks=%d, Placement=%s", first.configuration_name, first.method, first.solver, first.slice_set_name, len(commands),
                plan.nodes[0].placement)
    logger.info("  Internal SC lengths: per kick=%g m, total=%g m; first s=%g m, last s=%g m", first.sc_length,
                math.fsum(command.sc_length for command in commands), first.s, last.s)
    logger.info("  Internal SC aperture: Type=%s, Value=%s, Source=element", first.aperture_type, first.aperture_value)
    logger.info("  Internal SC output: Save field=%s, Save potential=%s, Save density=%s, "
                "Save turn ranges=%s", first.save_field, first.save_potential, first.save_density, first._save_turn_ranges)


def run_body_slices(element, beam, bunch, turn, transport, *, gpu=False):
    """Call transport(ds, on_center) for a whole bunch, then boundary SC.

    No reference clock advancement and no longitudinal rebinning occurs here.
    The owning element advances its reference clock once for the total length.
    Each SC entry point checks the aperture once before evaluating its source.
    The owning element checks again at its exit; other slice boundaries do not
    add aperture checks. SC also validates its field domain independently.
    """
    plan = element.slice_plan
    p = beam.particles
    region = slice(bunch.start_idx, bunch.end_idx)
    for i in range(plan.num_slices):
        entry_alive = p.tag[region] > 0
        pair = element._sc_nodes.get(i)
        callback = None
        if pair is not None:
            node, command = pair

            def callback(command=command):
                # Exclude upstream/local losses before deposition, preserving
                # first loss records even if an outer tracker also records loss.
                lost = entry_alive & (p.tag[region] <= 0)
                unrecorded = lost & (p.lost_turn[region] < 0)
                p.lost_position[region][unrecorded] = command.s
                p.lost_turn[region][unrecorded] = turn
                entry_alive[lost] = False
                if gpu:
                    command.apply_bunch_gpu(element._sc_sim, beam, bunch)
                else:
                    command.apply_bunch_cpu(element._sc_sim, beam, bunch)
                entry_alive[p.tag[region] <= 0] = False

        transport(plan.slice_length, callback if pair and node.placement == "center" else None)
        if pair and node.placement == "boundary":
            callback()
        lost = entry_alive & (p.tag[region] <= 0) & (p.lost_turn[region] < 0)
        p.lost_position[region][lost] = element.s - element.length + (i + 1) * plan.slice_length
        p.lost_turn[region][lost] = turn


def transport_with_center(advance, ds, on_center):
    """Adapt a complete short map (drift, matrix, solenoid) to a center hook."""
    if on_center is None:
        advance(ds)
    else:
        advance(ds / 2)
        on_center()
        advance(ds / 2)


_GPU_STAGE_HEADER = r"""
extern "C" __global__ void internal_stage(
    pass_particle_t* x,
    pass_particle_t* px,
    pass_particle_t* y,
    pass_particle_t* py,
    pass_particle_t* z,
    const pass_particle_t* dp,
    int* tag,
    float* lp,
    int* lt,
    int start,
    int end,
    pass_real_t beta0,
    pass_real_t reference_beta_gamma,
    pass_real_t invgamma,
    double L,
    pass_real_t s0,
    int turn,
    const pass_real_t* params,
    const pass_real_t* kn,
    const pass_real_t* ks,
    const pass_real_t* inv,
    int order,
    int action
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i >= end || tag[i] <= 0)
        return;
    pass_real_t xi = x[i], pxi = px[i], yi = y[i], pyi = py[i], zi = z[i], dpi = dp[i];
    int ti = tag[i];
    bool alive = true;
"""
_GPU_STAGE_FOOTER = r"""
    x[i]=xi;px[i]=pxi;y[i]=yi;py[i]=pyi;z[i]=zi;tag[i]=ti;
}
"""
_GPU_STAGE_MULTIPOLE = r"""
if (action != 2) {
    alive = pass_drift(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, L / 2, reference_beta_gamma, invgamma, s0, turn);
    if (alive)
        pass_kick(pxi, pyi, xi, yi, kn, ks, inv, order, L);
}
if (action != 1 && alive)
    pass_drift(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, L / 2, reference_beta_gamma, invgamma, s0, turn);
"""
_GPU_STAGE_SOLENOID = r"""
if (action == 3)
    sol_exact(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, L, params[0], beta0, reference_beta_gamma, s0, turn);
else {
    if (action != 2) {
        alive = sol_exact(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, L / 2, params[0], beta0, reference_beta_gamma, s0, turn);
        if (alive)
            sol_kick(pxi, pyi, xi, yi, kn, ks, inv, order, L);
    }
    if (action != 1 && alive)
        sol_exact(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, L / 2, params[0], beta0, reference_beta_gamma, s0, turn);
}
"""

_GPU_STAGE_BEND = r"""
pass_real_t h = params[0], k0 = params[1], momentum_ratio = 1 + dpi;
pass_real_t particle_beta_gamma = momentum_ratio * reference_beta_gamma,
            beta0_over_beta = beta0 * sqrt(1 + particle_beta_gamma * particle_beta_gamma) / particle_beta_gamma;
pass_real_t time_factor = sqrt(momentum_ratio * momentum_ratio + 1 / (reference_beta_gamma * reference_beta_gamma));
if (action == 3 || action == 4) {
    pass_real_t e = params[action == 3 ? 2 : 3], sn = sin(-e), cs = cos(-e);
    if (action == 3) {
        if (fabs(e) > PASS_EPS)
            alive = d_yrot(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, -e, sn, cs, beta0, time_factor, s0, turn);
        if (alive && fabs(k0) > PASS_EPS)
            alive = d_fringe(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, params[5], params[4], k0, beta0, time_factor, s0, turn);
        if (alive && fabs(e) > PASS_EPS)
            alive = d_wedge(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, -e, k0, sn, cs, beta0, beta0_over_beta, time_factor, s0, turn);
    } else {
        if (fabs(e) > PASS_EPS)
            alive = d_wedge(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, -e, k0, sn, cs, beta0, beta0_over_beta, time_factor, s0, turn);
        if (alive && fabs(k0) > PASS_EPS)
            alive = d_fringe(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, params[6], params[4], -k0, beta0, time_factor, s0, turn);
        if (alive && fabs(e) > PASS_EPS)
            alive = d_yrot(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, -e, sn, cs, beta0, time_factor, s0, turn);
    }
} else if (params[7] == 0) {
    if (action != 2) {
        alive = d_drift(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, L / 2, beta0_over_beta, reference_beta_gamma, s0, turn);
        if (alive)
            d_kick(pxi, zi, xi, dpi, L, h, k0, beta0, beta0_over_beta);
    }
    if (action != 1 && alive)
        d_drift(xi, pxi, yi, pyi, zi, dpi, ti, lp, lt, i, L / 2, beta0_over_beta, reference_beta_gamma, s0, turn);
} else {
    pass_real_t rho = fabs(h) > PASS_EPS ? 1 / h : 0;
    const double base = h * L / 4.0;
    const double z1 = pass_yoshida_z1, z0 = pass_yoshida_z0;
    pass_real_t sf, cf, shf, sm, cm, shm;
    d_polar_trig(base * z1, sf, cf, shf);
    d_polar_trig(base * (z1 + z0), sm, cm, shm);
    if (action != 2) {
        alive = d_rkr_drift(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, L / 2, h, k0, beta0, beta0_over_beta, reference_beta_gamma, rho, sf, cf, shf,
                            sm, cm, shm, s0, turn);
        if (alive)
            pxi -= L * k0 * h * xi;
    }
    if (action != 1 && alive)
        d_rkr_drift(xi, pxi, yi, zi, pyi, dpi, ti, lp, lt, i, L / 2, h, k0, beta0, beta0_over_beta, reference_beta_gamma, rho, sf, cf, shf, sm, cm,
                    shm, s0, turn);
}
"""


@lru_cache(maxsize=None)
def _stage_kernel(kind, dtype, device):
    import cupy as cp

    if kind == "multipole":
        from PASS.commands.element.multipole import (
            CUDA_REAL_PREAMBLE,
            MULTIPOLE_KERNEL_BODY,
        )

        source = CUDA_REAL_PREAMBLE + MULTIPOLE_KERNEL_BODY
        body = _GPU_STAGE_MULTIPOLE
    elif kind == "solenoid":
        from PASS.commands.element.solenoid import CUDA_REAL_PREAMBLE, SOLENOID_BODY

        source = CUDA_REAL_PREAMBLE + SOLENOID_BODY
        body = _GPU_STAGE_SOLENOID
    elif kind == "bend":
        from PASS.commands.element.dipole import CUDA_REAL_PREAMBLE, DIPOLE_BODY

        source = CUDA_REAL_PREAMBLE + DIPOLE_BODY
        body = _GPU_STAGE_BEND
    else:
        raise ValueError(f"unknown internal GPU transport {kind}")
    with cp.cuda.Device(device):
        particle_type = "float" if np.dtype(dtype).itemsize == 4 else "double"
        # Polar bend maps subtract a macroscopic radius from a small transverse
        # position. Keep those intermediates double even for float particles.
        return cp.RawKernel(
            f"typedef {particle_type} pass_particle_t;\n" + source + _GPU_STAGE_HEADER + body + _GPU_STAGE_FOOTER,
            "internal_stage",
            options=(
                "--std=c++17",
                f"-DPASS_USE_FLOAT={int(np.dtype(dtype).itemsize == 4 and kind != 'bend')}",
            ),
        )


def _dkd(launch, integrator, ds, on_center):
    weights = ((1.0, ) if integrator == "uniform" else (const.yoshida_z1, const.yoshida_z0, const.yoshida_z1))
    for index, weight in enumerate(weights):
        length = ds * weight
        if on_center is not None and index == len(weights) // 2:
            launch(length, 1)
            on_center()
            launch(length, 2)
        else:
            launch(length, 0)


def _prepare_body_coefficients(element):
    """Prepare fixed stage parameters once; bunch and particle state stays live."""
    coefficients = getattr(element, "_body_coefficients", None)
    if coefficients is not None:
        return coefficients
    name = type(element).__name__.lower()
    kn = ks = np.zeros(1)
    params = np.zeros(1)
    inv = np.ones(1)
    if name == "solenoid":
        params = np.array([element.ks])
        if element.has_multipoles:
            kn, ks, inv = element.kn, element.ksp, element.inv_fact
    elif name == "sbend":
        params = np.array([
            element.h,
            element.k0,
            element.e1,
            element.e2,
            element.hgap,
            element.fint,
            element.fintx,
            int(element.model == "rot-kick-rot"),
        ])
    elif name == "multipole":
        kn, ks, inv = element.kn, element.ks, element.inv_fact
    elif name == "kicker":
        kn, ks = np.array([-element.hk]), np.array([element.vk])
    elif name in ("quadrupole", "sextupole", "octupole"):
        order = {"quadrupole": 1, "sextupole": 2, "octupole": 3}[name]
        kn = np.zeros(order + 1)
        ks = kn.copy()
        kn[order] = getattr(element, f"k{order}")
        ks[order] = getattr(element, f"k{order}s")
        inv = np.array([1 / math.factorial(i) for i in range(order + 1)])
    errors = getattr(element, "field_errors", None)
    if errors is not None and errors.active and name in ("quadrupole", "sextupole", "octupole", "kicker"):
        kn, ks = errors.combine(kn * element.length, ks * element.length)
        kn, ks = kn / element.length, ks / element.length
        inv = np.ones(len(kn))
        for i in range(1, len(kn)):
            inv[i] = inv[i - 1] / i
    element._body_coefficients = (params, kn, ks, inv)
    return element._body_coefficients


def execute_element_body_gpu(element, sim):
    """Schedule shared device maps with field errors and optional internal SC.

    A center SC kick splits at the integrator's central kick, including the
    negative Yoshida stage; the positive SC integration weight is unchanged.
    """
    import cupy as cp

    from PASS.utils.aperture import check_aperture_gpu

    name = type(element).__name__.lower()
    if name == "elseparator":
        return element.execute_gpu(sim)  # Own finite-geometry map and SC scheduling.
    supported = {
        "drift",
        "quadrupole",
        "sextupole",
        "octupole",
        "multipole",
        "kicker",
        "solenoid",
        "sbend",
    }
    if name not in supported:
        raise RuntimeError(f"{name} GPU body transport is not yet supported")
    beam = sim.beams[element.beam_id]
    p = beam.particles
    real = p.real
    turn = sim.state.turn
    kind = ("solenoid" if name == "solenoid" else "bend" if name == "sbend" else "multipole")
    coefficients = _prepare_body_coefficients(element)
    errors = getattr(element, "field_errors", None)
    resources = getattr(element, "_gpu_body_resources", None)
    if resources is None:
        resources = element._gpu_body_resources = {}
    key = (np.dtype(p.dtype).str, cp.cuda.runtime.getDevice())
    if key not in resources:
        resources[key] = tuple(cp.asarray(a, dtype=np.float64 if kind == "bend" else p.dtype) for a in coefficients)
    cache = resources[key]
    order = np.int32(len(coefficients[1]) - 1)
    for bunch in beam.bunches:
        start, end = bunch.start_idx, bunch.end_idx
        n = end - start
        if n <= 0:
            continue
        blocks = ((n + 255) // 256, )
        threads = (256, )
        position = element.s - element.length

        def drift(length):
            from PASS.commands.element.drift import _get_transfer_drift_kernel

            _get_transfer_drift_kernel(p.dtype.str)(
                blocks,
                threads,
                (
                    p.x,
                    p.y,
                    p.z,
                    p.px,
                    p.py,
                    p.dp,
                    p.tag,
                    p.lost_position,
                    p.lost_turn,
                    np.int32(start),
                    np.int32(end),
                    real(1 / bunch.gamma**2),
                    real(length),
                    real(position),
                    np.int32(turn),
                ),
            )

        def launch_stage(length, action):
            stage_real = np.float64 if kind == "bend" else real
            _stage_kernel(kind, *key)(
                blocks,
                threads,
                (
                    p.x,
                    p.px,
                    p.y,
                    p.py,
                    p.z,
                    p.dp,
                    p.tag,
                    p.lost_position,
                    p.lost_turn,
                    np.int32(start),
                    np.int32(end),
                    stage_real(bunch.beta),
                    stage_real(bunch.beta * bunch.gamma),
                    stage_real(1 / bunch.gamma),
                    np.float64(length),
                    stage_real(position),
                    np.int32(turn),
                    *cache,
                    order,
                    np.int32(action),
                ),
            )

        def launch(length, action):
            if kind == "bend" and errors is not None and errors.active and action in (0, 1):
                launch_stage(length, 1)
                errors.kick_gpu(p, start, end, length / element.length)
                if action == 0:
                    launch_stage(length, 2)
            else:
                launch_stage(length, action)

        def matrix(length):
            kernel = _matrix_kernel(*key)
            kernel(
                blocks,
                threads,
                (
                    p.x,
                    p.px,
                    p.y,
                    p.py,
                    p.z,
                    p.dp,
                    p.tag,
                    np.int32(start),
                    np.int32(end),
                    real(bunch.beta),
                    real(bunch.beta * bunch.gamma),
                    real(1 / bunch.gamma),
                    real(length),
                    real(element.cos_theta),
                    real(element.sin_theta),
                    real(element.k_eff_base),
                    np.int32(1),
                ),
            )

        def transport(ds, on_center):
            nonlocal position
            end_position = position + ds
            position += ds / 2 if on_center is not None else ds
            callback = None
            if on_center is not None:

                def callback():
                    nonlocal position
                    on_center()
                    position = end_position

            if name == "drift":
                transport_with_center(drift, ds, callback)
            elif name == "quadrupole" and element.model == "mat-kick-mat":
                if errors is not None and errors.active:
                    from PASS.commands.element.error import _transport_matrix_errors
                    _transport_matrix_errors(matrix, lambda scale: errors.kick_gpu(p, start, end, scale), ds, element.length, element.integrator,
                                             callback)
                else:
                    transport_with_center(matrix, ds, callback)
            elif name == "solenoid" and not element.has_multipoles:
                transport_with_center(lambda length: launch(length, 3), ds, callback)
            else:
                _dkd(launch, element.integrator, ds, callback)
            position = end_position

        if name == "sbend":
            launch(0.0, 3)
        run_body_slices(element, beam, bunch, turn, transport, gpu=True)
        if name == "sbend":
            launch(0.0, 4)
        check_aperture_gpu(beam, bunch, element.aperture_type, element.aperture_value, element.s, turn)
        bunch.t0 += element.length / (bunch.beta * const.c)
    return True


@lru_cache(maxsize=None)
def _matrix_kernel(dtype, device):
    import cupy as cp

    from PASS.commands.element.quadrupole import CUDA_REAL_PREAMBLE, QUAD_MATRIX_BODY

    with cp.cuda.Device(device):
        return cp.RawKernel(
            CUDA_REAL_PREAMBLE + QUAD_MATRIX_BODY,
            "track_quadrupole_matrix",
            options=(
                "--std=c++17",
                f"-DPASS_USE_FLOAT={int(np.dtype(dtype).itemsize == 4)}",
            ),
        )
