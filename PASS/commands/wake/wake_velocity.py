"""Explicit source and witness velocity contracts for an integrated response."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityLaw:
    kind: str
    beta: float | None = None
    betas: tuple = ()
    source: tuple = ()
    witness: tuple = ()

    def __post_init__(self):
        if self.kind == "fixed":
            if self.beta is None or not np.isfinite(self.beta) or not 0 < self.beta <= 1:
                raise ValueError("Fixed response requires 0 < reference beta <= 1")
        elif self.kind == "factorized":
            for name in ("betas", "source", "witness"):
                object.__setattr__(self, name, tuple(getattr(self, name)))
            b = np.asarray(self.betas)
            if (len(b) < 2 or b[0] < 0 or b[-1] > 1 or np.any(np.diff(b) <= 0)
                    or len(self.source) != len(b) or len(self.witness) != len(b)
                    or not all(np.all(np.isfinite(v)) for v in (b, self.source, self.witness))):
                raise ValueError("Velocity coupling needs finite matching tables on increasing beta knots in [0, 1]")
        else:
            raise ValueError("Velocity law must be fixed or factorized")

    def validate(self, beta):
        b = np.asarray(beta)
        if np.any(~np.isfinite(b)) or np.any((b <= 0) | (b > 1)):
            raise ValueError("Wake beta must lie in (0, 1]")
        if self.kind == "fixed":
            if not np.all(np.isclose(b, self.beta, rtol=1e-12, atol=0)):
                raise ValueError(f"Fixed-beta response at beta={self.beta} cannot describe this trajectory; "
                                 "provide a model with source and witness velocity coupling")
        elif np.any((b < self.betas[0]) | (b > self.betas[-1])):
            raise ValueError("Particle beta is outside the supplied velocity-coupling table")

    def source_factor(self, beta):
        self.validate(beta)
        return np.ones_like(beta, dtype=float) if self.kind == "fixed" else np.interp(beta, self.betas, self.source)

    def witness_factor(self, beta):
        self.validate(beta)
        return np.ones_like(beta, dtype=float) if self.kind == "fixed" else np.interp(beta, self.betas, self.witness)


# ----------------------------------------------------------------------------
# GPU: source and witness velocity factors
# ----------------------------------------------------------------------------


def factor_gpu(component, betas, witness=False):
    import cupy as cp
    from .wake_models import device_arrays
    law = component.velocity
    if law is None or law.kind == "fixed":
        return 1.
    values = law.witness if witness else law.source
    if all(v == values[0] for v in values):
        return values[0]
    knots, values = device_arrays(law, "witness" if witness else "source", (law.betas, values))
    # Validity was checked using host reference beta before projection. Stored
    # source beta belongs to that historical passage, never today's reference.
    return cp.interp(betas, knots, values)



