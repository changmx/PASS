"""Portable host-side physical state for frozen and dynamic electron populations."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


def _validated_rng_state(values):
    """Copy and validate the PCG64 state used by both cloud models."""
    if not isinstance(values, Mapping) or set(values) != {"bit_generator", "state", "has_uint32", "uinteger"}:
        raise ValueError("electron cloud rng_state must be a PCG64 state")
    rng_state = deepcopy(dict(values))
    if rng_state["bit_generator"] != "PCG64" or not isinstance(rng_state["state"], Mapping) or set(rng_state["state"]) != {"state", "inc"}:
        raise ValueError("electron cloud rng_state must be a PCG64 state")
    for name, value, maximum in (
        ("state", rng_state["state"]["state"], 2**128 - 1),
        ("inc", rng_state["state"]["inc"], 2**128 - 1),
        ("has_uint32", rng_state["has_uint32"], 1),
        ("uinteger", rng_state["uinteger"], 2**32 - 1),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value <= maximum:
            raise ValueError(f"electron cloud RNG {name} is invalid")
    if rng_state["state"]["inc"] % 2 != 1:
        raise ValueError("electron cloud PCG64 increment must be odd")
    generator = np.random.PCG64(0)
    generator.state = rng_state
    return deepcopy(generator.state)


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"dynamic electron cloud {name} must be finite and numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"dynamic electron cloud {name} must be finite and numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"dynamic electron cloud {name} must be finite and numeric")
    return result


@dataclass(frozen=True)
class ElectronCloudState:
    """Coordinates in metres and positive numbers of represented electrons.

    ``source_length`` is the longitudinal length represented by the weights,
    not the machine length over which the beam receives a kick. Arrays stay
    on the host for portable snapshots and are copied to the selected solver.
    """

    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    source_length: float
    rng_state: dict

    def __post_init__(self):
        for name in ("x", "y", "weight"):
            raw = np.asarray(getattr(self, name))
            if raw.dtype.kind not in "fiu" or raw.ndim != 1:
                raise ValueError(f"electron cloud {name} must be a one-dimensional numeric array")
            values = np.array(raw, dtype=np.float64, copy=True)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"electron cloud {name} must be finite")
            values.flags.writeable = False
            object.__setattr__(self, name, values)
        if self.x.shape != self.y.shape or self.x.shape != self.weight.shape:
            raise ValueError("electron cloud x, y and weight shapes must match")
        if np.any(self.weight < 0) or not np.isfinite(self.weight.sum()):
            raise ValueError("electron cloud weights must be nonnegative with finite total")
        if isinstance(self.source_length, (bool, np.bool_)):
            raise ValueError("electron cloud source_length must be positive and finite")
        source_length = float(self.source_length)
        if not np.isfinite(source_length) or source_length <= 0:
            raise ValueError("electron cloud source_length must be positive and finite")
        object.__setattr__(self, "source_length", source_length)
        object.__setattr__(self, "rng_state", _validated_rng_state(self.rng_state))

    @property
    def n_macroparticles(self):
        return self.x.size

    def state_dict(self):
        """Return an independent JSON-compatible physical-state mapping."""
        return {
            "format": "PASS-frozen-electrons-1",
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "weight": self.weight.tolist(),
            "source_length": self.source_length,
            "rng_state": deepcopy(self.rng_state),
        }

    @classmethod
    def from_state_dict(cls, values):
        """Validate the complete snapshot before constructing a state."""
        if not isinstance(values, Mapping) or set(values) != {"format", "x", "y", "weight", "source_length", "rng_state"}:
            raise ValueError("invalid frozen electron-cloud state fields")
        if values["format"] != "PASS-frozen-electrons-1":
            raise ValueError("unsupported frozen electron-cloud state format")
        return cls(**{name: values[name] for name in ("x", "y", "weight", "source_length", "rng_state")})


@dataclass(frozen=True)
class DynamicElectronCloudState:
    """Physical electron coordinates and normalized mechanical momenta.

    ``x/y`` are metres, ``u = p/(m_e*c)`` is dimensionless, and ``weight``
    counts represented electrons over ``source_length`` metres. ``time`` is
    absolute physical time in seconds; a finite initial time may precede the
    first completed tracking turn. Empty arrays represent an absorbed cloud.
    """

    x: np.ndarray
    y: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    weight: np.ndarray
    source_length: float
    time: float | None
    rng_state: dict
    last_turn: int | None = None
    counters: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("x", "y", "ux", "uy", "uz", "weight"):
            raw = np.asarray(getattr(self, name))
            if raw.dtype.kind not in "fiu" or raw.ndim != 1:
                raise ValueError(f"dynamic electron cloud {name} must be a one-dimensional numeric array")
            values = np.array(raw, dtype=np.float64, copy=True)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"dynamic electron cloud {name} must be finite")
            values.flags.writeable = False
            object.__setattr__(self, name, values)
        if any(getattr(self, name).shape != self.x.shape for name in ("y", "ux", "uy", "uz", "weight")):
            raise ValueError("dynamic electron cloud position, momentum and weight shapes must match")
        if np.any(self.weight < 0) or not np.isfinite(self.weight.sum()):
            raise ValueError("dynamic electron cloud weights must be nonnegative with finite total")
        source_length = _finite_scalar(self.source_length, "source_length")
        if source_length <= 0:
            raise ValueError("dynamic electron cloud source_length must be positive and finite")
        object.__setattr__(self, "source_length", source_length)
        if self.time is not None:
            object.__setattr__(self, "time", _finite_scalar(self.time, "time"))
        if self.last_turn is not None:
            if isinstance(self.last_turn, (bool, np.bool_)) or not isinstance(self.last_turn, (int, np.integer)) or self.last_turn < 0:
                raise ValueError("dynamic electron cloud last_turn must be a nonnegative integer or None")
            if self.time is None:
                raise ValueError("dynamic electron cloud completed turns require a physical time")
            object.__setattr__(self, "last_turn", int(self.last_turn))
        object.__setattr__(self, "rng_state", _validated_rng_state(self.rng_state))
        if not isinstance(self.counters, Mapping):
            raise ValueError("dynamic electron cloud counters must be a mapping")
        counters = {}
        for name, value in self.counters.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError("dynamic electron cloud counter names must be nonempty without surrounding whitespace")
            _finite_scalar(value, f"counter {name!r}")
            counters[name] = int(value) if isinstance(value, (int, np.integer)) else float(value)
        object.__setattr__(self, "counters", counters)

    @property
    def n_macroparticles(self):
        return self.x.size

    def state_dict(self):
        """Copy the entire evolving source, clock, RNG and cumulative diagnostics."""
        values = {name: getattr(self, name).tolist() for name in ("x", "y", "ux", "uy", "uz", "weight")}
        values.update(format="PASS-dynamic-electrons-1",
                      source_length=self.source_length,
                      time=self.time,
                      last_turn=self.last_turn,
                      rng_state=deepcopy(self.rng_state),
                      counters=deepcopy(self.counters))
        return values

    @classmethod
    def from_state_dict(cls, values):
        """Validate all checkpoint fields without fixing population or total weight."""
        fields = {"x", "y", "ux", "uy", "uz", "weight", "source_length", "time", "rng_state", "last_turn", "counters"}
        if not isinstance(values, Mapping) or set(values) != {"format", *fields}:
            raise ValueError("invalid dynamic electron-cloud state fields")
        if values["format"] != "PASS-dynamic-electrons-1":
            raise ValueError("unsupported dynamic electron-cloud state format")
        return cls(**{name: values[name] for name in fields})
