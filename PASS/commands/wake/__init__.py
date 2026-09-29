"""Wake responses, source projection, solvers and per-location history.

Models, spectra and walls define responses. Components, velocity laws and
conventions define coupling; moments, timing and periodic projection define
sources. Solvers and convolution engines evaluate responses, while state and
execution manage history and resources. File I/O accepts canonical TFS only;
external numeric conversion belongs to PASS.tool.wake_conversion.
"""
from .wake_components import WakeComponent, SpatialTerm, COMPONENTS
from .convolution import ConvolutionGrid, PartitionedConvolution, ConvolutionState
from .time_convolution import TimeGrid, TimeConvolution, TimeConvolutionState
from .wake_models import WakeModel, ConstantWakeModel, ResonatorWakeModel, ResistiveWallWakeModel, TabulatedWakeModel
from .wake_spectrum import ImpedanceSpectrum, SpectrumWakeModel, RationalWakeModel, fit_spectrum
from .wake_velocity import VelocityLaw
from .wake_wall import RoundWallImpedance
from .wake_moments import WakeSourceProjector
from .wake_solvers import DirectSliceSolver, FFTConvolutionSolver, RecursiveResonatorSolver, RecursiveModalSolver
from .wake_state import WakeSources, WakeState
from .wake_io import WakeConvention, read_wake_file

__all__ = [
    "WakeComponent", "SpatialTerm", "ConvolutionGrid", "PartitionedConvolution", "ConvolutionState", "TimeGrid", "TimeConvolution",
    "TimeConvolutionState", "COMPONENTS", "WakeModel", "ConstantWakeModel", "ResonatorWakeModel", "ResistiveWallWakeModel", "TabulatedWakeModel",
    "ImpedanceSpectrum", "SpectrumWakeModel", "RationalWakeModel", "fit_spectrum", "VelocityLaw", "RoundWallImpedance", "RecursiveModalSolver",
    "WakeSourceProjector", "DirectSliceSolver", "FFTConvolutionSolver", "RecursiveResonatorSolver", "WakeSources", "WakeState", "WakeConvention",
    "read_wake_file"
]
