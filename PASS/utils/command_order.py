"""Command ordering metadata without importing tracking implementations."""

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
    "Bump": 300,
    "RFCavity": 300,
    "ElSeparator": 300,
    "Exciter": 300,
    "SpaceCharge": 400,
    "WakeField": 500,
    "BeamBeam": 600,
    "ElectronCloud": 700,
    "LumiMonitor": 800,
    "PhaseAdvanceMonitor": 800,
    "DistMonitor": 800,
    "StatMonitor": 800,
    "ParticleMonitor": 800,
    "Other": 999,
}


def command_priority(command_type: str) -> int:
    """Return the same-s sorting priority for a PASS command type."""
    return COMMAND_PRIORITY.get(command_type, COMMAND_PRIORITY["Other"])
