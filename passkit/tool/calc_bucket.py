import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from typing import Literal


def resolve_kinematics(Ek_per_nucleon, E0_per_nucleon, A, eta=None, gamma_t=None):
    """
    Calculate basic parameters

    Parameters:
        Ek_per_nucleon: kinetic energy per nucleon (eV)
        E0_per_nucleon: rest mass per nucleon (eV), 0.511e6 (e), 938.272e6 (p), 931.494e6 (u)
        A: mass number 
        eta: phase slip factor
        gammat: transition gamma


    Returns:
        Es: total energy (multiplied by mass number A) 
        gamma: relativistic factor
        beta: relativistic speed
        eta: phase slip factor
    """

    E_tot_per_nucleon = E0_per_nucleon + Ek_per_nucleon
    Es = A * E_tot_per_nucleon
    gamma = E_tot_per_nucleon / E0_per_nucleon
    beta = np.sqrt(1.0 - 1.0 / (gamma**2))

    if eta is not None:
        calc_eta = eta
    elif gamma_t is not None:
        calc_eta = (1.0 / (gamma_t**2)) - (1.0 / (gamma**2))
    else:
        raise ValueError("You must provide either 'eta' or 'gamma_t'.")

    return Es, gamma, beta, calc_eta


def calc_bucket_width(phi_s_deg, Ek_per_nucleon=None, E0_per_nucleon=None, A=None, eta=None, gamma_t=None):
    """
    Calculate bucket width

    Parameters:
        phi_s_deg: synchronous phase (degree)
        Ek_per_nucleon: kinetic energy per nucleon (eV)
        E0_per_nucleon: rest mass per nucleon (eV), 0.511e6 (e), 938.272e6 (p), 931.494e6 (u)
        A: mass number 
        eta: phase slip factor
        gammat: transition gamma

    Returns:
        dict
    """
    if eta is None and gamma_t is not None:
        if None in (Ek_per_nucleon, E0_per_nucleon, A):
            raise ValueError("To calculate eta from gamma_t, kinetic parameters (Ek, E0, A) must be provided.")
        _, _, _, calc_eta = resolve_kinematics(Ek_per_nucleon, E0_per_nucleon, A, eta=None, gamma_t=gamma_t)
    elif eta is not None:
        calc_eta = eta
    else:
        raise ValueError("Provide either 'eta' or 'gamma_t'.")

    phi_s = np.radians(phi_s_deg)
    phi_1 = np.pi - phi_s

    def potential_diff(phi):
        if calc_eta < 0:
            return np.cos(phi) + np.cos(phi_s) - (np.pi - phi_s - phi) * np.sin(phi_s)
        else:
            return -np.cos(phi) - np.cos(phi_s) + (np.pi - phi_s - phi) * np.sin(phi_s)

    try:
        phi_2 = brentq(potential_diff, -np.pi, phi_s)
    except ValueError:
        phi_2 = brentq(potential_diff, -2 * np.pi, phi_s)

    width_rad = phi_1 - phi_2

    return {
        'phi_left_deg': np.degrees(phi_2),
        'phi_right_deg': np.degrees(phi_1),
        'width_rad': width_rad,
        'width_deg': np.degrees(width_rad),
        'eta_used': calc_eta
    }


def calc_bucket_height(V0, phi_s_deg, h, Ek_per_nucleon, E0_per_nucleon, A, q=1, eta=None, gamma_t=None):
    """
    Calculate bucket height

    Parameters:
        V0: RF voltage (V)
        phi_s_deg: synchronous phase (degree)
        h: harmonic number
        Ek_per_nucleon: kinetic energy per nucleon (eV)
        E0_per_nucleon: rest mass per nucleon (eV), 0.511e6 (e), 938.272e6 (p), 931.494e6 (u)
        A: mass number
        q: charge number
        eta: phase slip factor
        gammat: transition gamma

    Returns:
        dict
    """
    Es, gamma, beta, calc_eta = resolve_kinematics(Ek_per_nucleon, E0_per_nucleon, A, eta, gamma_t)
    phi_s = np.radians(phi_s_deg)

    shape_factor = np.sqrt(2 * np.cos(phi_s) - (np.pi - 2 * phi_s) * np.sin(phi_s))
    scaling_coeff = np.sqrt((q * V0) / (np.pi * h * np.abs(calc_eta) * (beta**2) * Es))

    dp_p_max = scaling_coeff * shape_factor
    dE_max_MeV = ((beta**2) * Es * dp_p_max) / 1e6

    return {'dp_p_max': dp_p_max, 'dE_max_MeV': dE_max_MeV, 'shape_factor': shape_factor, 'eta_used': calc_eta}


def compute_bucket_data(V0, phi_s_deg, h, R0, Ek_per_nucleon, E0_per_nucleon, A, q=1, eta=None, gamma_t=None, num_orbits=6):
    """
    Model: Computes the 2D Hamiltonian field and separatrix for the RF bucket.
    Automatically handles relativistic kinematics and transition energy physics.

    Parameters:
        V0: RF voltage
        phi_s_deg: synchronous phase in degrees
        h: harmonic number
        R0: radius
        Ek_per_nucleon: kinetic energy per nucleon (eV)
        E0_per_nucleon: rest mass per nucleon (eV), 0.511e6 (e), 938.272e6 (p), 931.494e6 (u)
        A: mass number of the particle
        q: charge number of the particle
        eta: phase slip factor, eta = 1/gammat**2 - 1/gamma**2
        gammat: transition gamma
        num_orbits: the desired number of stable orbits to be calculated

    Returns:
        dict
    """
    Es, gamma, beta, calc_eta = resolve_kinematics(Ek_per_nucleon, E0_per_nucleon, A, eta, gamma_t)
    phi_s = np.radians(phi_s_deg)

    phi_min = phi_s - 1.2 * np.pi
    phi_max = phi_s + 1.2 * np.pi
    Phi = np.linspace(phi_min, phi_max, 1000)

    coeff_dp = np.sqrt(q * V0 / (np.pi * h * np.abs(calc_eta) * (beta**2) * Es))
    dp_max = coeff_dp * 1.5
    DP = np.linspace(-dp_max, dp_max, 1000)

    Phi_grid, DP_grid = np.meshgrid(Phi, DP)

    if calc_eta < 0:
        R_grid = np.cos(Phi_grid) + np.cos(phi_s) - (np.pi - phi_s - Phi_grid) * np.sin(phi_s)
        R_max = 2 * np.cos(phi_s) - (np.pi - 2 * phi_s) * np.sin(phi_s)
    else:
        R_grid = -np.cos(Phi_grid) - np.cos(phi_s) + (np.pi - phi_s - Phi_grid) * np.sin(phi_s)
        R_max = -2 * np.cos(phi_s) + (np.pi - 2 * phi_s) * np.sin(phi_s)

    H = (DP_grid / coeff_dp)**2 - R_grid

    return {
        'phi_s_rad': phi_s,
        'phi_s_deg': phi_s_deg,
        'Phi_grid': Phi_grid,
        'DP_grid': DP_grid,
        'H': H,
        'H_sep': 0.0,
        'levels_inside': np.linspace(-R_max * 0.95, -abs(-R_max) * 0.02, num_orbits),
        'Es': Es,
        'beta': beta,
        'gamma': gamma,
        'eta': calc_eta,
        'R0': R0,
        'h': h
    }


if __name__ == "__main__":
    pass
