"""Compatibility imports for BeamBeam schemas, now independent of tracking."""

from PASS.para.schema.beam_beam import (
    BeamBeamConfig,
    BeamBeamConfiguration,
    BeamBeamItem,
    BeamBeamLuminosityConfig,
    BeamBeamSourceConfig,
    FrozenParameters,
    _CollisionInput,
    _validate_sequences,
    _value,
    load_beam_beam,
)

__all__ = [
    "BeamBeamConfig",
    "BeamBeamConfiguration",
    "BeamBeamItem",
    "BeamBeamLuminosityConfig",
    "BeamBeamSourceConfig",
    "FrozenParameters",
    "load_beam_beam",
]
