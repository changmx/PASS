"""Schema definitions for PASS simulation parameters.

All parameter models use pydantic v2 BaseModel with alias fields.
The alias is the JSON key consumed by the PASS engine (case-insensitive
after convert_keys_to_lower). The Python attribute name is clean.

Models:
    MainConfig       — global simulation parameters
    BunchConfig      — per-bunch injection parameters
    OffsetConfig     — injection offset (x or y)
    InjectionItem    — injection sequence node
    TwissItem       — twiss transport point
    ElementBase      — base for all magnet/element types
    DriftItem ... — specific element types
    StatMonitorItem ...  — monitor types
    Sequence         — ordered container + sort
    SpaceChargeConfig — space-charge parameters
"""

from PASS.para.schema.main import MainConfig, TimingConfig
from PASS.para.schema.rf import RFComponent, ReferenceClock
from PASS.para.schema.bunch import BunchConfig, OffsetConfig, ScanGridConfig, InjectionItem
from PASS.para.schema.twiss import TwissItem
from PASS.para.schema.slicer import SlicerItem
from PASS.para.schema.transverse_feedback import TransversePickupItem, TransverseFeedbackItem, design_feedback_fir
from PASS.para.schema.wake_field import (
    WakeFieldItem,
    WakeFieldConfig,
    WakeResourceConfig,
    WakeComponentConfig,
    ConstantWake,
    ResonatorWake,
    ResistiveWallWake,
    TabulatedWake,
    UltrarelativisticWallWake,
    ImpedanceWake,
    FittedImpedanceWake,
    ModalWake,
    WakeVelocity,
    WakeSolverGroup,
    WakeSpatialTerm,
    WakeConvolutionGrid,
    WakeTimeGrid,
    FileWake,
    WakeFileConvention,
)
from PASS.para.schema.elements import (
    ElementBase,
    DriftItem,
    MarkerItem,
    SBendItem,
    QuadrupoleItem,
    SextupoleItem,
    OctupoleItem,
    MultipoleItem,
    SolenoidItem,
    ElectronBeamConfig,
    ElectronCoolerItem,
    KickerItem,
    BumpItem,
    ElSeparatorItem,
    ExciterItem,
    RFCavityItem,
    ReorganizeBunchItem,
    CrossingAngleItem,
    CrabCavityItem,
    FloatWaisterItem,
)
from PASS.para.schema.monitors import StatMonitorItem, DistMonitorItem, PhaseAdvanceMonitorItem, ParticleMonitorItem
from PASS.para.schema.slow_extraction import SlowExtractionItem, SlowExtractionMonitorItem
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.space_charge import SpaceChargeConfig, SpaceChargeResourceConfig, SpaceChargeItem, ElementSpaceCharge
from PASS.para.schema.electron_cloud import ElectronCloudBuildUpConfiguration, ElectronCloudConfig, ElectronCloudConfiguration, ElectronCloudItem
from PASS.para.schema.ibs import IBSConfig, IBSConfiguration, IBSItem, IBSOpticsConfig
from PASS.commands.collision.config import BeamBeamConfig, BeamBeamConfiguration, BeamBeamSourceConfig, FrozenParameters, BeamBeamItem

__all__ = [
    'MainConfig', 'TimingConfig', 'BunchConfig', 'OffsetConfig', 'ScanGridConfig', 'InjectionItem', 'TwissItem', 'SlicerItem', 'TransversePickupItem',
    'TransverseFeedbackItem', 'design_feedback_fir', 'WakeFieldItem', 'WakeFieldConfig', 'WakeResourceConfig', 'WakeComponentConfig', 'ConstantWake',
    'ResonatorWake', 'ResistiveWallWake', 'TabulatedWake', 'UltrarelativisticWallWake', 'ImpedanceWake', 'FittedImpedanceWake', 'ModalWake',
    'WakeVelocity', 'WakeSolverGroup', 'WakeSpatialTerm', 'WakeConvolutionGrid', 'WakeTimeGrid', 'FileWake', 'WakeFileConvention', 'ElementBase',
    'DriftItem', 'MarkerItem', 'SBendItem', 'QuadrupoleItem', 'SextupoleItem', 'OctupoleItem', 'MultipoleItem', 'SolenoidItem', 'ElectronBeamConfig',
    'ElectronCoolerItem', 'KickerItem', 'BumpItem', 'ElSeparatorItem', 'ExciterItem', 'RFCavityItem', 'RFComponent', 'ReferenceClock',
    'ReorganizeBunchItem', 'StatMonitorItem', 'DistMonitorItem', 'PhaseAdvanceMonitorItem', 'ParticleMonitorItem', 'Sequence', 'SpaceChargeConfig',
    'SpaceChargeResourceConfig', 'SpaceChargeItem', 'ElementSpaceCharge', 'BeamBeamConfig', 'BeamBeamConfiguration', 'BeamBeamSourceConfig',
    'FrozenParameters', 'BeamBeamItem', 'CrossingAngleItem', 'CrabCavityItem', 'FloatWaisterItem', 'ElectronCloudConfig',
    'ElectronCloudConfiguration', 'ElectronCloudItem', 'ElectronCloudBuildUpConfiguration', 'IBSConfig', 'IBSConfiguration', 'IBSItem',
    'IBSOpticsConfig', 'SlowExtractionItem', 'SlowExtractionMonitorItem'
]
