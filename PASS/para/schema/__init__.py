"""Schema definitions for PASS simulation parameters.

All parameter models use pydantic v2 BaseModel with alias fields.
The alias is the JSON key consumed by the PASS engine (case-insensitive
after convert_keys_to_lower). The Python attribute name is clean.

Models:
    MainConfig       — global simulation parameters
    BunchConfig      — per-bunch injection parameters
    OffsetConfig     — injection offset (x or y)
    InjectionItem    — injection sequence node
    TwissPoint       — twiss transport point
    ElementBase      — base for all magnet/element types
    DriftElement ... — specific element types
    StatMonitor ...  — monitor types
    Sequence         — ordered container + sort
    SpaceChargeConfig — space-charge parameters
"""

from PASS.para.schema.main import MainConfig, TimingConfig
from PASS.para.schema.bunch import BunchConfig, OffsetConfig, InjectionItem
from PASS.para.schema.twiss import TwissPoint
from PASS.para.schema.slicer import Slicer
from PASS.para.schema.elements import (
    ElementBase,
    DriftElement,
    MarkerElement,
    SBendElement,
    QuadrupoleElement,
    SextupoleElement,
    OctupoleElement,
    MultipoleElement,
    SolenoidElement,
    KickerElement,
    ElSeparatorElement,
    ExciterElement,
    RFCavityElement,
    ReorganizeBunchElement,
)
from PASS.para.schema.monitors import (
    StatMonitor,
    DistMonitor,
    PhaseMonitor,
    ParticleMonitor,
)
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.space_charge import SpaceChargeConfig

__all__ = [
    "MainConfig",
    "TimingConfig",
    "BunchConfig",
    "OffsetConfig",
    "InjectionItem",
    "TwissPoint",
    "Slicer",
    "ElementBase",
    "DriftElement",
    "MarkerElement",
    "SBendElement",
    "QuadrupoleElement",
    "SextupoleElement",
    "OctupoleElement",
    "MultipoleElement",
    "SolenoidElement",
    "KickerElement",
    "ElSeparatorElement",
    "ExciterElement",
    "RFCavityElement",
    "ReorganizeBunchElement",
    "StatMonitor",
    "DistMonitor",
    "PhaseMonitor",
    "ParticleMonitor",
    "Sequence",
    "SpaceChargeConfig",
]
