"""Sequence container: ordered collection of schema items.

Items can be any pydantic BaseModel with a ``command`` field
(TwissPoint, ElementBase subclasses, monitors, InjectionItem, etc.).

The sequence dict key is the item name; the item itself does not store its name.
On export, each item is serialized via model_dump(by_alias=True) and the
result is sorted by (s, command priority) via toolkit.sort_sequence.
"""

from collections import OrderedDict
from pydantic import BaseModel

from PASS.para.toolkit import sort_sequence


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
        return sort_sequence(raw)
