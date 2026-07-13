"""Shared utilities for the PASS parameter system.

Contents:
    - class_map: MADX keyword → PASS Command registry name
    - sort_sequence: sort sequence dict by (s, command priority)
"""

from collections import OrderedDict


# MADX KEYWORD → PASS Command registry name (lowercase, matches @Command.register)
class_map = {
    "marker": "marker",
    "drift": "drift",
    "sbend": "sbend",
    "rbend": "sbend",          # MADX rbend maps to SBend
    "quadrupole": "quadrupole",
    "sextupole": "sextupole",
    "octupole": "octupole",
    "multipole": "multipole",
    "solenoid": "solenoid",
    "hkicker": "kicker",
    "vkicker": "kicker",
    "kicker": "kicker",
    "tkicker": "kicker",
    "monitor": "drift",        # BPM treated as drift
    "elseparator": "elseparator",
    "exciter": "exciter",
    "rfcavity": "rfcavity",
}


# Command priority for same-s sorting (lower = earlier)
_COMMAND_PRIORITY = {
    "Injection": 0,
    "SortBunch": 100,
    "Twiss": 200,
    "Marker": 300,
    "Drift": 300,
    "SBend": 300,
    "Quadrupole": 300,
    "Sextupole": 300,
    "Octupole": 300,
    "Multipole": 300,
    "Solenoid": 300,
    "Kicker": 300,
    "RF": 300,
    "ElSeparator": 300,
    "Exciter": 300,
    "SpaceCharge": 400,
    "WakeField": 500,
    "BeamBeam": 600,
    "ElectronCloud": 700,
    "LumiMonitor": 800,
    "PhaseMonitor": 800,
    "DistMonitor": 800,
    "StatMonitor": 800,
    "ParticleMonitor": 800,
    "Other": 999,
}


def _convert_ordereddict(obj):
    """Recursively convert OrderedDict → plain dict for JSON serialization."""
    if isinstance(obj, OrderedDict):
        return {k: _convert_ordereddict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_ordereddict(item) for item in obj]
    else:
        return obj


def sort_sequence(sequence: dict) -> dict:
    """Sort sequence items by (S position, command priority).

    Args:
        sequence: {name: {\"S (m)\": float, \"Command\": str, ...}}

    Returns:
        Plain dict sorted by (s, priority).
    """
    sorted_seq = OrderedDict(
        sorted(
            sequence.items(),
            key=lambda item: (
                item[1]["S (m)"],
                _COMMAND_PRIORITY.get(item[1].get("Command", ""), 999),
            ),
        )
    )
    return _convert_ordereddict(sorted_seq)
