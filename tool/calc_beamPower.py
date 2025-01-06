import numpy as np


def calc_beam_average_power(
    m_static,  # statistic mass of electron/proton/nucleon
    Ek,  # kinetic energy of electron/proton/nucleon
    N,  # particles per pulse
    repetition_frequency,
    N_particle=1,
    N_charge=1,
    is_print=True,
    **kwargs
):
    """
    对于电子/质子来说, N_particle=1, N_charge=1
    对于离子来说, N_particle=离子中的核子数目, N_charge=离子中的电子数目
    可选参数有: E_total、gamma、beta
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

    T = 1
    I_avg = N * N_charge * e * repetition_frequency / T

    # P = W/T = N*Ek*e/T = N*Ek*(e/T), for proton e/T = I, for ion e/T is not equal to I but I/N_charge
    # Ek * e means transfer erergy from eV to J
    P = N * N_particle * Ek * e * repetition_frequency / T

    if is_print:
        print(
            "Ek: {0:.6f} MeV/u, I_average: {1:.4f} uA, P: {2:.4f} kW".format(
                Ek / 1e6, I_avg * 1e6, P / 1e3
            )
        )


def calc_beam_peak_power(
    m_static,  # statistic mass of electron/proton/nucleon
    Ek,  # kinetic energy of electron/proton/nucleon
    N,  # particles per pulse
    T,
    N_particle=1,
    N_charge=1,
    is_print=True,
    **kwargs
):
    """
    对于电子/质子来说, N_particle=1, N_charge=1
    对于离子来说, N_particle=离子中的核子数目, N_charge=离子中的电子数目
    可选参数有: E_total、gamma、beta
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

    # I_peak = N * N_charge * e / T

    # P = W/T = N*Ek*e/T = N*Ek*(e/T), for proton e/T = I, for ion e/T is not equal to I but I/N_charge
    # Ek * e means transfer erergy from eV to J
    P = N * N_particle * Ek * e / T

    if is_print:
        print("Ek: {0:.6f} MeV/u, P: {1:.4f} MW".format(Ek / 1e6, P / 1e6))


if __name__ == "__main__":
    m_proton = 938.27208816e6
    m_electron = 0.510998950e6
    m_unified_atomic_mass = 931.4941024236e6
    m_muon = 105.6583755e6
    m_pion = 139.57039e6

    calc_beam_average_power(m_proton, 1.0e9, 1.5e13, 200, N_particle=1, N_charge=1)
    calc_beam_average_power(
        m_unified_atomic_mass, 300e6, 1.0e12, 200, N_particle=4, N_charge=2
    )

    calc_beam_peak_power(
        m_proton, 1.4e9, 8e11, 150e-9, N_particle=1, N_charge=1
    )  # bunch length = 150 ns
    calc_beam_peak_power(
        m_unified_atomic_mass, 500e6, 1e10, 150e-9, N_particle=12, N_charge=6
    )  # bunch length = 150 ns
    calc_beam_peak_power(
        m_unified_atomic_mass, 60e6, 5e8, 150e-9, N_particle=209, N_charge=32
    )  # bunch length = 150 ns
    calc_beam_peak_power(
        m_unified_atomic_mass, 60e6, 5e8, 150e-9, N_particle=238, N_charge=37
    )  # bunch length = 150 ns
