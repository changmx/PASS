"""Private state of one physical wake location and one beam."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class WakeSources:
    times: np.ndarray
    widths: np.ndarray
    moments: dict
    betas: np.ndarray | None = None

    def __post_init__(self):
        times = np.array(self.times, dtype=float, copy=True)
        widths = np.array(self.widths, dtype=float, copy=True)
        if (times.ndim != 1 or widths.shape != times.shape or np.any(widths < 0)
                or not np.all(np.isfinite(times)) or not np.all(np.isfinite(widths))):
            raise ValueError("Wake source times/widths must be finite matching 1D arrays, widths >= 0")
        moments = {key: np.array(value, dtype=float, copy=True) for key, value in self.moments.items()}
        betas = np.ones_like(times) if self.betas is None else np.array(self.betas, dtype=float, copy=True)
        if betas.shape != times.shape or np.any(~np.isfinite(betas)) or np.any((betas <= 0) | (betas > 1)):
            raise ValueError("Wake source betas must match the bins and lie in (0, 1]")
        if any(v.shape != times.shape or not np.all(np.isfinite(v)) for v in moments.values()):
            raise ValueError("Wake source moments must be finite and match the time bins")
        for array in (times, widths, betas, *moments.values()):
            array.flags.writeable = False
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "widths", widths)
        object.__setattr__(self, "moments", moments)
        object.__setattr__(self, "betas", betas)


@dataclass
class WakeState:
    history: deque = field(default_factory=deque)
    mode_amplitudes: dict = field(default_factory=dict)
    last_time: float | None = None
    last_turn: int | None = None
    last_source_end: float | None = None
    convolution: object = None

    def fork(self):
        return WakeState(deque(self.history), {k: v.copy() for k, v in self.mode_amplitudes.items()},
                         self.last_time, self.last_turn, self.last_source_end, self.convolution)

    def reset(self):
        self.history.clear()
        self.mode_amplitudes.clear()
        self.last_time = self.last_turn = self.last_source_end = None
        self.convolution = None

    def state_dict(self):
        def host(v):
            return v.get() if hasattr(v, "get") else np.asarray(v)
        return {"history": [{"turn": turn, "times": host(s.times).tolist(), "widths": host(s.widths).tolist(),
                             "betas": host(s.betas).tolist(),
                             "moments": [{"powers": list(k), "values": host(v).tolist()} for k, v in s.moments.items()]}
                            for turn, s in self.history],
                "modes": [{"component": k, "real": host(v).real.tolist(), "imag": host(v).imag.tolist(),
                           "complex": v.dtype.kind == "c"} for k, v in self.mode_amplitudes.items()],
                "last_time": self.last_time, "last_turn": self.last_turn, "last_source_end": self.last_source_end,
                "convolution": (self.convolution.state_dict() if hasattr(self.convolution, "state_dict")
                                else self.convolution)}

    @classmethod
    def from_state_dict(cls, data):
        history = deque()
        for row in data["history"]:
            history.append((row["turn"], WakeSources(row["times"], row["widths"],
                {tuple(m["powers"]): m["values"] for m in row["moments"]}, row["betas"])))
        modes = {}
        for row in data["modes"]:
            value = np.asarray(row["real"], float)
            if row["complex"]:
                value = value+1j*np.asarray(row["imag"], float)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError("Wake checkpoint contains invalid mode state")
            modes[row["component"]] = value
        for name in ("last_time", "last_source_end"):
            if data[name] is not None and not np.isfinite(data[name]):
                raise ValueError("Wake checkpoint contains invalid physical time")
        last_turn = data["last_turn"]
        if last_turn is not None and (isinstance(last_turn, bool) or not isinstance(last_turn, int) or last_turn < 0):
            raise ValueError("Wake checkpoint contains invalid turn")
        if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 or last_turn is None or t > last_turn for t, _ in history):
            raise ValueError("Wake checkpoint history turn is inconsistent")
        import copy
        return cls(history, modes, data["last_time"], last_turn, data["last_source_end"],
                   copy.deepcopy(data.get("convolution")))


# ----------------------------------------------------------------------------
# GPU: device source arrays
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceSources:
    times: object
    widths: object
    moments: dict
    betas: object
    point: bool = False
    grid: tuple | None = None

    @classmethod
    def upload(cls, source):
        if isinstance(source, cls):
            return source
        import cupy as cp
        return cls(cp.asarray(source.times), cp.asarray(source.widths),
                   {k: cp.asarray(v) for k, v in source.moments.items()},
                   cp.asarray(source.betas), bool(np.all(source.widths == 0)))

    def ordered(self, order):
        return DeviceSources(self.times[order], self.widths[order],
            {k: v[order] for k, v in self.moments.items()}, self.betas[order], self.point)
