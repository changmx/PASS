def calc_beam_average_power(m_static, Ek, N, repetition_frequency, N_particle=1, N_charge=1, is_print=True, **kwargs):
    """Print average kinetic beam power in kW and signed current in microamps.

    Ek is the sole energy input: eV per nucleon for ions, or eV per particle
    with N_particle=1 for electrons/protons. N_particle is the nucleon count
    for ions, N_charge is the signed charge number, N is particles per pulse,
    and repetition_frequency is in Hz. m_static is retained for call compatibility.
    E_total, gamma and beta keywords raise TypeError; convert them to Ek first.
    Printed energy uses MeV/u, with /u meaning per particle when N_particle=1.
    is_print=False suppresses output; the function returns None in either case.
    """
    if any(name in kwargs for name in ("E_total", "gamma", "beta")):
        raise TypeError("calc_beam_average_power uses Ek only; convert E_total, gamma or beta to Ek before calling")
    e = 1.602176634 * 1e-19
    I_avg = N * N_charge * e * repetition_frequency
    # N_particle * Ek * e is the kinetic energy per complete particle in joules.
    P = N * N_particle * Ek * e * repetition_frequency

    if is_print:
        print("Ek: {0:.6f} MeV/u, I_average: {1:.4f} uA, P: {2:.4f} kW".format(Ek / 1e6, I_avg * 1e6, P / 1e3))


def calc_beam_peak_power(m_static, Ek, N, T, N_particle=1, N_charge=1, is_print=True, **kwargs):
    """Print pulse-averaged kinetic beam power in MW; peak for a flat pulse.

    Ek is the sole energy input: eV per nucleon for ions, or eV per particle
    with N_particle=1 for electrons/protons. N_particle is the nucleon count
    for ions, N is particles per pulse, and T is the pulse duration in seconds.
    m_static and N_charge are retained for call compatibility; neither enters
    this kinetic-power calculation. E_total, gamma and beta keywords raise
    TypeError; convert them to Ek first. Printed energy uses MeV/u, with /u
    meaning per particle when N_particle=1. is_print=False suppresses output;
    the function returns None in either case.
    """
    if any(name in kwargs for name in ("E_total", "gamma", "beta")):
        raise TypeError("calc_beam_peak_power uses Ek only; convert E_total, gamma or beta to Ek before calling")
    e = 1.602176634 * 1e-19
    # N_particle * Ek * e is the kinetic energy per complete particle in joules.
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
    calc_beam_average_power(m_unified_atomic_mass, 300e6, 1.0e12, 200, N_particle=4, N_charge=2)

    calc_beam_peak_power(m_proton, 1.4e9, 8e11, 150e-9, N_particle=1, N_charge=1)  # bunch length = 150 ns
    calc_beam_peak_power(m_unified_atomic_mass, 500e6, 1e10, 150e-9, N_particle=12, N_charge=6)  # bunch length = 150 ns
    calc_beam_peak_power(m_unified_atomic_mass, 60e6, 5e8, 150e-9, N_particle=209, N_charge=32)  # bunch length = 150 ns
    calc_beam_peak_power(m_unified_atomic_mass, 60e6, 5e8, 150e-9, N_particle=238, N_charge=37)  # bunch length = 150 ns
