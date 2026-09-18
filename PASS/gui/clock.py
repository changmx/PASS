"""Resolve a portable prescribed-clock snapshot without allocating particles."""
import math

from PASS.para.schema.rf import ReferenceClock
from PASS.utils.constants import const


def reference_clock_snapshot(data):
    if data.get("Reference clock") is not None:
        return ReferenceClock.model_validate(data["Reference clock"]).model_dump(by_alias=True)
    injection = next((v for v in data.get("Sequence", {}).values() if isinstance(v, dict) and v.get("Command") == "Injection"), None)
    if injection is None:
        raise ValueError("默认时钟需要 Injection")
    bunch = next((v for k, v in injection.items() if k.startswith("bunch") and isinstance(v, dict) and v.get("Harmonic ID of this bunch", 0) == 0),
                 None)
    if bunch is None:
        raise ValueError("默认时钟需要 harmonic ID=0 的束团")
    # Match BunchInfo's tracking mass convention, not the independent Tools catalog.
    protons, neutrons = data["Number of Protons"], data["Number of Neutrons"]
    mass = const.m_e_eV if protons == neutrons == 0 else const.m_p_eV if (protons, neutrons) == (1, 0) else const.m_u_eV
    energy = float(bunch["Kinetic Energy per Nucleon (eV/u)"])
    circumference = float(data["Circumference (m)"])
    if not math.isfinite(energy) or energy <= 0 or not math.isfinite(circumference) or circumference <= 0:
        raise ValueError("默认时钟需要正的有限动能与环长")
    gamma = 1.0 + energy / mass
    beta = math.sqrt(1.0 - 1.0 / gamma / gamma)
    return ReferenceClock(frequency=beta * const.c / circumference).model_dump(by_alias=True)
