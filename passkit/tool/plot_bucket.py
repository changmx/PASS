import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from typing import Literal

from passkit.tool.calc_bucket import resolve_kinematics, calc_bucket_width, calc_bucket_height, compute_bucket_data


def plot_bucket_on_ax(ax, data, x_axis: Literal['z', 'phi'] = 'z', y_axis: Literal['dp', 'dE'] = 'dp', phi_unit: Literal['deg', 'rad'] = 'rad'):
    """
    Plot bucket area on a given matplotlib axis.
    """
    Phi_grid = data['Phi_grid']
    DP_grid = data['DP_grid']

    if x_axis == 'z':
        X_plot = -(Phi_grid - data['phi_s_rad']) * (data['R0'] / data['h'])
        x_label = 'Longitudinal Position $z$ (m)'
        x_center = 0.0
    elif x_axis == 'phi':
        if phi_unit == 'deg':
            X_plot = np.degrees(Phi_grid)
            x_label = 'RF Phase $\\phi$ (deg)'
            x_center = data['phi_s_deg']
        elif phi_unit == 'rad':
            X_plot = Phi_grid
            x_label = 'RF Phase $\\phi$ (rad)'
            x_center = data['phi_s_rad']
        else:
            raise ValueError("phi_unit must be 'deg' or 'rad'")
    else:
        raise ValueError("x_axis must be 'z' or 'phi'")

    if y_axis == 'dp':
        Y_plot = DP_grid
        y_label = 'Momentum Deviation $\\Delta p/p_0$'
        y_center = 0.0
    elif y_axis == 'dE':
        Y_plot = ((data['beta']**2) * data['Es'] * DP_grid) / 1e6
        y_label = 'Energy Deviation $\\Delta E$ (MeV)'
        y_center = 0.0
    else:
        raise ValueError("y_axis must be 'dp' or 'dE'")

    ax.contour(X_plot, Y_plot, data['H'], levels=data['levels_inside'], colors='teal', linewidths=1.5, alpha=0.8)
    ax.contour(X_plot, Y_plot, data['H'], levels=[data['H_sep']], colors='crimson', linewidths=2.5)
    ax.plot(x_center, y_center, 'yo', markersize=8, markeredgecolor='black')

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.axhline(y_center, color='black', linestyle=':', alpha=0.4)
    ax.axvline(x_center, color='black', linestyle=':', alpha=0.4)
    ax.grid(True, linestyle='--', alpha=0.4)


def plot_example1():
    """
    Compare with the results of Accelerator Physics (Fourth edition, S.Y. Lee, P 237, Fig 3.3)
    """
    ion_params = {
        'V0': 100e3,
        'phi_s_deg': 30.0,
        'h': 1,
        'R0': 40.0,
        'Ek_per_nucleon': 45e6,
        'E0_per_nucleon': 938.272e6,
        'A': 1,
        'q': 1,
        'gamma_t': np.sqrt(1 / 0.0434),
        'num_orbits': 4
    }

    w_data = calc_bucket_width(phi_s_deg=ion_params['phi_s_deg'],
                               Ek_per_nucleon=ion_params['Ek_per_nucleon'],
                               E0_per_nucleon=ion_params['E0_per_nucleon'],
                               A=ion_params['A'],
                               gamma_t=ion_params['gamma_t'])

    h_data = calc_bucket_height(V0=ion_params['V0'],
                                phi_s_deg=ion_params['phi_s_deg'],
                                h=ion_params['h'],
                                Ek_per_nucleon=ion_params['Ek_per_nucleon'],
                                E0_per_nucleon=ion_params['E0_per_nucleon'],
                                A=ion_params['A'],
                                q=ion_params['q'],
                                gamma_t=ion_params['gamma_t'])

    # 1. Compute physics data ONCE
    bucket_data = compute_bucket_data(**ion_params)

    print(f"Bucket widt h: {w_data['width_deg']:.2f} deg ({w_data['width_rad']:.2f} rad)")
    print(f"Bucket height: max dp/p = {h_data['dp_p_max']:.5e}, max dE = {h_data['dE_max_MeV']:.2f} MeV\n")
    print(f"Calculated gamma: {bucket_data['gamma']:.4f}")
    print(f"Calculated beta : {bucket_data['beta']:.4f}")
    print(f"Calculated eta  : {bucket_data['eta']:.6f}")

    # 2. Setup a 2x2 matrix dashboard
    fig, ax = plt.subplots()
    # fig.suptitle(f"Heavy Ion RF Bucket Matrix ($\\gamma_t={heavy_ion_params['gamma_t']}$)", fontsize=16, fontweight='bold')

    ax.set_title("[phi - dp/p]")
    plot_bucket_on_ax(ax, bucket_data, x_axis='phi', y_axis='dp', phi_unit="rad")

    ax.set_xlim(-1, np.pi)
    ax.set_ylim(-0.02, 0.02)
    plt.show()


# ==================== Test / Execution Block ====================
if __name__ == "__main__":
    plot_example1()
