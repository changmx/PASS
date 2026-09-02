COMMAND_PRIORITY = {
    "Injection": 0,
    "SortBunch": 100,
    "ReorganizeBunch": 150,
    "Slicer": 160,
    "Twiss": 200,
    "Marker": 300,
    "Drift": 300,
    "SBend": 300,
    "RBend": 300,
    "Quadrupole": 300,
    "Sextupole": 300,
    "Octupole": 300,
    "Multipole": 300,
    "Solenoid": 300,
    "Kicker": 300,
    "RFCavity": 300,
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


def command_priority(command_type: str) -> int:
    """Return the same-s sorting priority for a PASS command type."""
    return COMMAND_PRIORITY.get(command_type, COMMAND_PRIORITY["Other"])


from .command import Command
from .twiss import Twiss
from .injection import Injection
from .reorganize import ReorganizeBunch
from .sort_bunch import SortBunch
from .slicer import Slicer, SliceSet
from .element.marker import Marker
from .element.drift import Drift
from .element.dipole import SBend
from .element.quadrupole import Quadrupole
from .element.sextupole import Sextupole
from .element.octupole import Octupole
from .element.multipole import Multipole
from .element.solenoid import Solenoid
from .element.kicker import Kicker
from .element.elseparator import ElSeparator
from .element.exciter import Exciter
from .element.rfcavity import RFCavity
from .monitor.statistic import StatMonitor
from .monitor.distribution import DistMonitor
from .monitor.particle_monitor import ParticleMonitor

__all__ = [
    "COMMAND_PRIORITY",
    "command_priority",
    "Command",
    "Twiss",
    "Injection",
    "SortBunch",
    "ReorganizeBunch",
    "Slicer",
    "SliceSet",
    "Marker",
    "Drift",
    "SBend",
    "Quadrupole",
    "Sextupole",
    "Octupole",
    "Multipole",
    "Solenoid",
    "Kicker",
    "ElSeparator",
    "Exciter",
    "RFCavity",
    "StatMonitor",
    "DistMonitor",
    "ParticleMonitor",
]
