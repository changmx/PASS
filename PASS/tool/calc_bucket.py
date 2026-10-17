import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import cosdg, sindg


def _get_bucket_potential(phi_s_deg):
    """Return the geometric potential, assuming the supplied phase is stable."""
    if not np.isfinite(phi_s_deg):
        raise ValueError("RF bucket requires a finite synchronous phase")
    phase_deg = math.remainder(phi_s_deg, 360.0)
    if abs(phase_deg) == 90.0:
        raise ValueError("RF bucket requires a non-degenerate synchronous phase (cos(phi_s) != 0)")
    # Degree-based trigonometry preserves small distances from 90/270 degrees.
    sin_phase, cos_phase = sindg(phase_deg), cosdg(phase_deg)
    direction = 1.0 if sin_phase * cos_phase >= 0 else -1.0
    # Mirror each stable branch so its lower saddle lies at a positive offset.
    saddle_offset = 2 * math.atan2(abs(cos_phase), abs(sin_phase))

    def _potential(offset):
        offset = np.asarray(offset)
        # Retain the cubic term for narrow buckets without subtractive cancellation.
        cubic = offset**3 * (1 / 6 - offset**2 / 120 + offset**4 / 5040)
        offset_minus_sin = np.where(np.abs(offset) < 1e-2, cubic, offset - np.sin(offset))
        return 2 * abs(cos_phase) * np.sin(offset / 2)**2 - abs(sin_phase) * offset_minus_sin

    barrier = float(_potential(saddle_offset))
    if not np.isfinite(barrier) or barrier <= 0:
        raise ValueError("RF bucket has no numerically resolved positive barrier")
    return cos_phase, direction, saddle_offset, barrier, _potential


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
    Calculate geometric bucket boundaries, assuming the supplied phase is stable.

    Charge and voltage are not inputs, so this function does not validate physical
    stability. The boundaries enclose the original, unreduced synchronous phase.
    eta is resolved and returned for compatibility; it does not set the geometry.

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

    _, direction, saddle_offset, barrier, potential = _get_bucket_potential(phi_s_deg)
    left, right = -saddle_offset, saddle_offset
    if saddle_offset != np.pi:
        # The reflected lower saddle brackets the other turning point.
        fraction = brentq(lambda value: potential(-saddle_offset * value) - barrier,
                          0.0,
                          1.0,
                          xtol=4 * np.finfo(float).eps,
                          rtol=4 * np.finfo(float).eps)
        left *= fraction
    if direction < 0:
        left, right = -right, -left
    width_rad = right - left

    return {
        'phi_left_deg': phi_s_deg + np.degrees(left),
        'phi_right_deg': phi_s_deg + np.degrees(right),
        'width_rad': width_rad,
        'width_deg': np.degrees(width_rad),
        'eta_used': calc_eta
    }


def calc_bucket_height(V0, phi_s_deg, h, Ek_per_nucleon, E0_per_nucleon, A, q=1, eta=None, gamma_t=None):
    """
    Calculate the bucket height for a stable physical RF phase.

    Uses the frozen single-harmonic, small-relative-momentum model. Signed q and
    V0 must satisfy -q*V0*eta*cos(phi_s) > 0. shape_factor is the square root of
    the lower dimensionless barrier of the two adjacent unstable fixed points.

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
    cos_phase, _, _, barrier, _ = _get_bucket_potential(phi_s_deg)
    if not all(np.isfinite(value) for value in (q, V0, calc_eta, h, beta, Es)) or h <= 0 or beta <= 0 or Es <= 0:
        raise ValueError("RF bucket requires finite parameters and positive h, beta and total energy")
    if not -q * V0 * calc_eta * cos_phase > 0:
        raise ValueError("RF bucket requires -q*V0*eta*cos(phi_s) > 0 at the physical RF phase")
    shape_factor = np.sqrt(barrier)
    scaling_coeff = np.sqrt(abs(q * V0) / (np.pi * h * abs(calc_eta) * beta**2 * Es))

    dp_p_max = scaling_coeff * shape_factor
    dE_max_MeV = ((beta**2) * Es * dp_p_max) / 1e6

    return {'dp_p_max': dp_p_max, 'dE_max_MeV': dE_max_MeV, 'shape_factor': shape_factor, 'eta_used': calc_eta}


def compute_bucket_data(V0, phi_s_deg, h, R0, Ek_per_nucleon, E0_per_nucleon, A, q=1, eta=None, gamma_t=None, num_orbits=6):
    """
    Compute the 2D Hamiltonian and separatrix in the frozen single-harmonic,
    small-relative-momentum model. Signed q and V0 must satisfy
    -q*V0*eta*cos(phi_s) > 0. The grid resolves the bucket around the original
    synchronous phase; H_sep remains zero and interior contour levels negative.

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
    cos_phase, direction, saddle_offset, barrier, potential = _get_bucket_potential(phi_s_deg)
    if not all(np.isfinite(value) for value in (q, V0, calc_eta, h, beta, Es)) or h <= 0 or beta <= 0 or Es <= 0:
        raise ValueError("RF bucket requires finite parameters and positive h, beta and total energy")
    if not -q * V0 * calc_eta * cos_phase > 0:
        raise ValueError("RF bucket requires -q*V0*eta*cos(phi_s) > 0 at the physical RF phase")

    # Scale both axes with the bucket so narrow, valid buckets remain resolved.
    offsets = np.linspace(-1.2 * saddle_offset, 1.2 * saddle_offset, 1000)
    Phi = phi_s + offsets
    coeff_dp = np.sqrt(abs(q * V0) / (np.pi * h * abs(calc_eta) * beta**2 * Es))
    dp_max = 1.1 * coeff_dp * np.sqrt(barrier)
    DP = np.linspace(-dp_max, dp_max, 1000)

    Phi_grid, DP_grid = np.meshgrid(Phi, DP)

    H = (DP_grid / coeff_dp)**2 + potential(direction * offsets)[None, :] - barrier

    return {
        'phi_s_rad': phi_s,
        'phi_s_deg': phi_s_deg,
        'Phi_grid': Phi_grid,
        'DP_grid': DP_grid,
        'H': H,
        'H_sep': 0.0,
        'levels_inside': np.linspace(-barrier * 0.95, -barrier * 0.02, num_orbits),
        'Es': Es,
        'beta': beta,
        'gamma': gamma,
        'eta': calc_eta,
        'R0': R0,
        'h': h
    }


if __name__ == "__main__":
    pass
