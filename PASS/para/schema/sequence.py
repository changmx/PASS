"""Sequence container: ordered collection of schema items.

Items can be any pydantic BaseModel with a ``command`` field
(TwissPoint, ElementBase subclasses, monitors, InjectionItem, etc.).

The sequence dict key is the item name; the item itself does not store its name.
On export, each item is serialized via model_dump(by_alias=True) and the
result is sorted by (s, command priority).
"""

from collections import OrderedDict
from pydantic import BaseModel


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


def _sort_sequence(sequence: dict) -> dict:
    """Sort sequence items by (S position, command priority).

    Args:
        sequence: {name: {"S (m)": float, "Command": str, ...}}

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


class Sequence:
    """Ordered container for sequence items.

    Usage::

        seq = Sequence()
        seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))
        seq.add("qd1", QuadrupoleElement(s=1.0, k1l=0.2, ...))
        seq.add("stat1", StatMonitor(s=0.0))

        # export to engine-compatible dict
        raw = seq.to_dict()
    """

    def __init__(self):
        self._items: OrderedDict[str, BaseModel] = OrderedDict()

    def add(self, name: str, item: BaseModel) -> "Sequence":
        """Add a named item. Returns self for chaining."""
        self._items[name] = item
        return self

    def add_many(self, items: dict[str, BaseModel]) -> "Sequence":
        """Add multiple named items from a dict."""
        self._items.update(items)
        return self

    def remove(self, name: str) -> "Sequence":
        """Remove an item by name."""
        self._items.pop(name, None)
        return self

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items.items())

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __getitem__(self, name: str) -> BaseModel:
        return self._items[name]

    def names(self) -> list[str]:
        return list(self._items.keys())

    def to_dict(self) -> dict:
        """Convert to engine-compatible dict (sorted by s, priority).

        For items that have a custom serialization (e.g. InjectionItem
        with bunch0/bunch1 keys), their to_sequence_dict() is used if available.
        """
        raw = {}
        for name, item in self._items.items():
            if hasattr(item, "to_sequence_dict"):
                raw[name] = item.to_sequence_dict()
            else:
                raw[name] = item.model_dump(by_alias=True)
        return _sort_sequence(raw)
