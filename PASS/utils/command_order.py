"""Command ordering metadata without importing tracking implementations."""

from collections.abc import Mapping
import math
from numbers import Integral

from PASS.utils.constants import const

COMMAND_PRIORITY = {
    "Injection": 0,
    "SortBunch": 100,
    "ReorganizeBunch": 150,
    "Slicer": 160,
    "Twiss": 200,
    "Marker": 300,
    "Drift": 300,
    "SBend": 300,
    "RBend": 300,
    "Quadrupole": 300,
    "Sextupole": 300,
    "Octupole": 300,
    "Multipole": 300,
    "Solenoid": 300,
    "ElectronCooler": 300,
    "Kicker": 300,
    "Bump": 300,
    "RFCavity": 300,
    "ElSeparator": 300,
    "Exciter": 300,
    "CrossingAngle": 300,
    "CrabCavity": 300,
    "FloatWaister": 300,
    "SpaceCharge": 400,
    "IBS": 450,
    "WakeField": 500,
    "BeamBeam": 600,
    "ElectronCloud": 700,
    "LumiMonitor": 800,
    "PhaseAdvanceMonitor": 800,
    "DistMonitor": 800,
    "StatMonitor": 800,
    "ParticleMonitor": 800,
    "SlowExtraction": 850,
    "SlowExtractionMonitor": 860,
    "Other": 999,
}


def command_priority(command_type: str) -> int:
    """Return the same-s sorting priority for a PASS command type."""
    if command_type in COMMAND_PRIORITY:
        return COMMAND_PRIORITY[command_type]
    # Command.create accepts case-insensitive command names; raw-input sorting
    # must use the same priority as the corresponding constructed command.
    name = str(command_type).casefold()
    return next((priority for kind, priority in COMMAND_PRIORITY.items() if kind.casefold() == name), COMMAND_PRIORITY["Other"])


def _command_value(command, attribute, alias, default=None):
    if isinstance(command, Mapping):
        return command.get(alias, command.get(alias.lower(), command.get(attribute, default)))
    return getattr(command, attribute, default)


def command_position_key(command, eps=None):
    """Use one position tolerance for runtime, generation and validation."""
    position = float(_command_value(command, "s", "S (m)", 0.0))
    if not math.isfinite(position):
        raise ValueError("Command S (m) must be finite")
    eps = const.eps if eps is None else float(eps)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("Command position tolerance must be finite and nonnegative")
    scaled = position / eps if eps else position
    return round(scaled) if eps and math.isfinite(scaled) else position


def command_sort_key(command, eps=None):
    """Return the common position and explicit Order or legacy priority."""
    order = _command_value(command, "order", "Order")
    if order is not None and (isinstance(order, bool) or not isinstance(order, Integral)):
        raise ValueError("Command Order must be an integer, not a coerced number or boolean")
    kind = _command_value(command, "command", "Command", getattr(command, "cmd_type", "Other"))
    return command_position_key(command, eps), int(order) if order is not None else command_priority(kind)


def sort_commands(items, *, key=None, eps=None):
    """Sort stably; a position with any Order requires distinct Order on all nodes."""
    items = list(items)
    extract = key if key is not None else lambda item: item
    groups = {}
    for item in items:
        command = extract(item)
        position, priority = command_sort_key(command, eps)
        groups.setdefault(position, []).append((_command_value(command, "order", "Order"), command))
    for group in groups.values():
        orders = [order for order, command in group]
        if not any(order is not None for order in orders):
            continue
        position = _command_value(group[0][1], "s", "S (m)", 0.0)
        if any(order is None for order in orders):
            raise ValueError(f"Every command at S={position} must specify Order when any command there uses Order")
        if len(set(orders)) != len(orders):
            raise ValueError(f"Commands at S={position} must have distinct Order values")
    return sorted(items, key=lambda item: command_sort_key(extract(item), eps))
