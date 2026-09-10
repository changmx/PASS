"""Shared body-slicing utilities and midpoint space-charge scheduling.

Machine positions use the element exit coordinate (the MAD-X S convention).
Yoshida's signed internal stages never determine the SC integration weight.
"""

from dataclasses import dataclass
import logging
import math
from numbers import Integral

from PASS.para.schema.space_charge import parse_element_space_charge, validate_loss_aperture
from PASS.utils.aperture import check_aperture_cpu
from PASS.utils.constants import const

logger = logging.getLogger(__name__)


def resolve_internal_sc_aperture(config, parent_type, parent_value, name, sim, beam_id):
    """Use the element aperture for both deposition walls and particle losses."""
    kind = str(parent_type).strip().lower()
    dimensions = validate_loss_aperture(kind, parent_value)
    if kind == "default":
        # Generic element default is +/-1 m, not the SC grid rectangle.
        kind, dimensions = "rectangle", [1.0, 1.0]
    elif kind == "off":
        dimensions = []
    if config.aperture_type != "default" and (
        config.aperture_type != kind or config.aperture_value != dimensions
    ):
        key = (beam_id, name, config.aperture_type, repr(config.aperture_value), kind, repr(dimensions))
        warned = getattr(sim, "_internal_sc_aperture_warnings", None)
        if warned is None:
            warned = sim._internal_sc_aperture_warnings = set()
        if key not in warned:
            logger.warning("Element %r internal SC aperture %s %s differs from element aperture %s %s; "
                           "using the element aperture", name, config.aperture_type,
                           config.aperture_value, kind, dimensions)
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
    nodes = tuple(SCNode(j, j * m + (m - 1) // 2, placement,
                         (j + 0.5) * sc_length, sc_length) for j in range(num_kicks))
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
    config = parse_element_space_charge(raw)
    config = resolve_internal_sc_aperture(config, element.aperture_type, element.aperture_value,
                                          element.cmd_name, sim, element.beam_id)
    element.slice_plan = make_slice_plan(element.length, requested, config.num_kicks)
    if not math.isfinite(element.s):
        raise ValueError("Element S (m) must be finite")
    # Existing PIC and analytic SC implementations are CPU-only. Fail before
    # any particle transport rather than silently omitting collective kicks.
    if getattr(sim.cfg, "use_gpu", False) or str(getattr(sim.cfg, "backend", "cpu")).lower() == "gpu":
        raise RuntimeError("Internal SpaceCharge GPU execution is not implemented; use backend='cpu'")
    from PASS.commands.space_charge import SpaceCharge
    command_values = config.model_dump(by_alias=True)
    command_values.pop("Num kicks")
    for node in element.slice_plan.nodes:
        command = SpaceCharge(element.beam_id, sim, **command_values,
            **{"Name": element.cmd_name, "S (m)": element.s - element.length + node.offset,
               "SC start (m)": element.s - element.length + node.offset - node.length / 2,
               "SC length (m)": node.length})
        command.parent_element = element.cmd_name
        command.internal_node_index = node.index
        element._sc_nodes[node.slice_index] = (node, command)
    element._sc_sim = sim
    logger.info("%s: external slices requested=%d actual=%d, SC kicks=%d, placement=%s, SC length=%g m",
                element.cmd_name, requested, element.slice_plan.num_slices, config.num_kicks,
                element.slice_plan.nodes[0].placement, element.slice_plan.nodes[0].length)


def print_element_slicing(element):
    """Report the effective plan and SC aperture from element.print()."""
    plan = element.slice_plan
    logger.info("  Slicing: requested=%d, actual=%d, external slice length=%g m",
                plan.requested_slices, plan.num_slices, plan.slice_length)
    if not element._sc_nodes:
        status = "disabled by top-level Space charge.Enabled" if element._sc_requested else "off"
        logger.info("  Internal SC: %s", status)
        return
    commands = [command for _, command in element._sc_nodes.values()]
    first, last = commands[0], commands[-1]
    logger.info("  Internal SC: Configuration=%s, Method=%s, Solver=%s, Slice set=%s, "
                "Num kicks=%d, Placement=%s",
                first.configuration_name, first.method, first.solver, first.slice_set_name,
                len(commands), plan.nodes[0].placement)
    logger.info("  Internal SC lengths: per kick=%g m, total=%g m; first s=%g m, last s=%g m",
                first.sc_length, math.fsum(command.sc_length for command in commands), first.s, last.s)
    logger.info("  Internal SC aperture: Type=%s, Value=%s, Source=element",
                first.aperture_type, first.aperture_value)
    logger.info("  Internal SC output: Save field=%s, Save potential=%s, Save density=%s, "
                "Save turn ranges=%s",
                first.save_field, first.save_potential, first.save_density, first._save_turn_ranges)


def guard_internal_sc_gpu(element):
    if element._sc_nodes:
        raise RuntimeError("Internal SpaceCharge GPU execution is not implemented; use backend='cpu'")


def run_body_slices(element, beam, bunch, turn, transport):
    """Call transport(ds, on_center) for a whole bunch, then boundary SC.

    No reference clock advancement and no longitudinal rebinning occurs here.
    The owning element advances its reference clock once for the total length.
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
                check_aperture_cpu(beam, bunch, element.aperture_type,
                                   element.aperture_value, command.s, turn)
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
