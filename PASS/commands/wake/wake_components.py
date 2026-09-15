"""Source/test monomials, independent of the temporal wake model."""
from dataclasses import dataclass


# Name: (kick plane, source powers (x,y), witness powers (x,y)).
COMPONENTS = {
    "longitudinal": ("z", (0, 0), (0, 0)),
    "constant_x": ("x", (0, 0), (0, 0)),
    "constant_y": ("y", (0, 0), (0, 0)),
    "dipolar_x": ("x", (1, 0), (0, 0)),
    "dipolar_y": ("y", (0, 1), (0, 0)),
    "dipolar_xy": ("x", (0, 1), (0, 0)),
    "dipolar_yx": ("y", (1, 0), (0, 0)),
    "quadrupolar_x": ("x", (0, 0), (1, 0)),
    "quadrupolar_y": ("y", (0, 0), (0, 1)),
    "quadrupolar_xy": ("x", (0, 0), (0, 1)),
    "quadrupolar_yx": ("y", (0, 0), (1, 0)),
}


@dataclass(frozen=True)
class SpatialTerm:
    """Charge-weighted source monomial and witness kick monomial, in metres."""
    plane: str
    source_powers: tuple = (0, 0)
    test_powers: tuple = (0, 0)

    def __post_init__(self):
        if self.plane not in {"x", "y", "z"}:
            raise ValueError("Wake kick plane must be x, y or z")
        for name in ("source_powers", "test_powers"):
            values = tuple(getattr(self, name))
            if len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 2147483647 for v in values):
                raise ValueError("Spatial powers must be two nonnegative int32 integers")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class WakeComponent:
    kind: str
    model: object
    scale: float = 1.0
    velocity: object = None
    spatial: SpatialTerm | None = None

    def __post_init__(self):
        if self.kind not in COMPONENTS and self.kind != "custom":
            raise ValueError(f"Unknown wake component {self.kind!r}")
        if (self.kind == "custom") != (self.spatial is not None):
            raise ValueError("Only custom components require an explicit SpatialTerm")
        import math
        if not math.isfinite(self.scale):
            raise ValueError("Wake component scale must be finite")
        if getattr(self.model, "round_pipe", False) and self.kind not in {
            "longitudinal", "dipolar_x", "dipolar_y"
        }:
            raise ValueError("Round resistive wall supports longitudinal and diagonal dipolar components")

    @property
    def plane(self):
        return self.spatial.plane if self.spatial is not None else COMPONENTS[self.kind][0]

    @property
    def source_powers(self):
        return self.spatial.source_powers if self.spatial is not None else COMPONENTS[self.kind][1]

    @property
    def test_powers(self):
        return self.spatial.test_powers if self.spatial is not None else COMPONENTS[self.kind][2]

    @property
    def longitudinal(self):
        return self.plane == "z"

    def source_moment(self, source):
        moment = source.moments[self.source_powers]
        return moment if self.velocity is None else moment*self.velocity.source_factor(source.betas)

    def witness_factor(self, betas):
        return 1. if self.velocity is None else self.velocity.witness_factor(betas)


# ----------------------------------------------------------------------------
# GPU: coupled source moments
# ----------------------------------------------------------------------------


def moment_gpu(component, source):
    from .wake_velocity import apply_factor_gpu
    moment=source.moments[component.source_powers]
    law=component.velocity
    if law is None or law.kind=='fixed' or all(v==1. for v in law.source):
        return moment
    return apply_factor_gpu(component,source.betas,moment)


def moments_gpu(components, source):
    """Channel rows; a single channel is a view, not a device copy."""
    import cupy as cp
    rows = [moment_gpu(c, source) for c in components]
    return rows[0][None, :] if len(rows) == 1 else cp.stack(rows)
