from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class _PhysicalConstants:
    """
    Physical constants (SI units unless noted).
    Values from Particle Data Group 2026 (https://pdg.lbl.gov/2026/reviews/contents_sports.html).
    """
    c: Final[float] = 299792458.0  # speed of light (m/s)
    pi: Final[float] = 3.141592653589793
    e: Final[float] = 1.602176634e-19  # elementary charge (C)

    eps: Final[float] = 1e-10  # small tolerance

    # Electron mass
    m_e_eV: Final[float] = 0.51099895000e6  # eV/c^2
    m_e_kg: Final[float] = 9.1093837015e-31  # kg

    # Proton mass
    m_p_eV: Final[float] = 938.27208816e6  # eV/c^2
    m_p_kg: Final[float] = 1.67262192369e-27  # kg

    # Unified atomic mass unit (u), (mass 12C atom)/12
    m_u_eV: Final[float] = 931.49410242e6  # eV/c^2
    m_u_kg: Final[float] = 1.66053906660e-27  # kg

    # Neutron mass
    m_n_eV: Final[float] = 939.56542052e6  # eV/c^2
    m_n_kg: Final[float] = 1.00866491595 * m_u_kg  # kg

    # Deuteron
    m_d_eV: Final[float] = 1875.61294257e6  # eV/c^2

    # Permittivity and permeability of free space
    epsilon0: Final[float] = 8.8541878128e-12  # F/m
    mu0: Final[float] = 1.00000000055 * 4 * pi * 1e-7  # exact (N/A^2)

    # Classical radii
    r_e: Final[float] = 2.8179403262e-15  # electron (m)
    r_p: Final[float] = 1.5346982672e-18  # proton (m), rp = re * me / mp

    h: Final[float] = 6.62607015e-34  # Planck constant (J·s)
    h_bar: Final[float] = h / (2 * pi)  # Planck constant, reduced (J·s)


const = _PhysicalConstants()
