import numpy as np


def calc_magnetic_rigidity(
    m_static, N_particle=1, N_charge=1, is_print=False, **kwargs
):
    """
    输入参数静止质量为必须, 可选参数有: E_total、gamma、beta
    """
    c = 299792458
    e = 1.602176634 * 1e-19

    E_total = 0
    gamma = 0
    beta = 0

    if "E_total" in kwargs:
        E_total = kwargs["E_total"]
        gamma = E_total / m_static
        beta = np.sqrt(1 - 1 / gamma**2)
    elif "gamma" in kwargs:
        gamma = kwargs["gamma"]
        E_total = gamma * m_static * N_particle
        beta = np.sqrt(1 - 1 / gamma**2)
    elif "beta" in kwargs:
        beta = kwargs["beta"]
        gamma = np.sqrt(1 / (1 - beta**2))
        E_total = gamma * m_static * N_particle

    p_eV = (
        gamma * m_static * N_particle * beta * c
    )  # The units of p_eV and m are eV/c and eV/c^2.
    p_j = p_eV / c / c * e  # The unit of p_j is joule.
    Brho = p_j / (e * N_charge)

    if is_print:
        print(
            "Brho: {0:.6f}, E total (GeV): {1:.3f}, beta: {2:.9f}, gamma: {3:.3f}".format(
                Brho, E_total / 1e9, beta, gamma
            )
        )

    return Brho


def derive_from_magnetic_rigidity(m_static, is_print=False, **kwargs):
    pass


if __name__ == "__main__":
    m_proton = 938.27208816e6
    m_electron = 0.510998950e6
    m_unified_atomic_mass = 931.4941024236e6
    m_muon = 105.6583755e6
    m_pion = 139.57039e6

    calc_magnetic_rigidity(m_proton, E_total=25.8e9, is_print=True)
    calc_magnetic_rigidity(m_proton, E_total=1e9 + m_proton, is_print=True)
    # calc_magnetic_rigidity(m_muon, E_total=E_muon, is_print=True)
    # calc_magnetic_rigidity(m_pion, E_total=E_pion, is_print=True)
    calc_magnetic_rigidity(m_electron, E_total=3.5e9, is_print=True)
    calc_magnetic_rigidity(m_electron, E_total=5e9, is_print=True)

    calc_magnetic_rigidity(m_proton, E_total=19.08e9 + m_proton, is_print=True)
    calc_magnetic_rigidity(m_electron, E_total=3.5e9 + m_electron, is_print=True)

    calc_magnetic_rigidity(
        m_unified_atomic_mass,
        E_total=25e6 + m_unified_atomic_mass,
        N_particle=9,
        N_charge=4,
        is_print=True,
    )
    calc_magnetic_rigidity(
        m_unified_atomic_mass,
        E_total=45e6 + m_unified_atomic_mass,
        N_particle=9,
        N_charge=4,
        is_print=True,
    )
    calc_magnetic_rigidity(
        m_unified_atomic_mass,
        E_total=650e6 + m_unified_atomic_mass,
        N_particle=9,
        N_charge=4,
        is_print=True,
    )

    calc_magnetic_rigidity(m_proton, E_total=100e6 + m_proton, is_print=True)
    calc_magnetic_rigidity(
        m_unified_atomic_mass,
        E_total=500e6 + m_unified_atomic_mass,
        N_particle=12,
        N_charge=6,
        is_print=True,
    )
    calc_magnetic_rigidity(
        m_unified_atomic_mass,
        E_total=50e6 + m_unified_atomic_mass,
        N_particle=238,
        N_charge=33,
        is_print=True,
    )
    calc_magnetic_rigidity(
        m_proton,
        E_total=1400e6 + m_proton,
        N_particle=1,
        N_charge=1,
        is_print=True,
    )
    calc_magnetic_rigidity(
        m_unified_atomic_mass,
        E_total=4e6 + m_unified_atomic_mass,
        N_particle=12,
        N_charge=6,
        is_print=True,
    )
