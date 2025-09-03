import numpy as np
import matplotlib.pyplot as plt

epsilon0 = 8.8541878128e-12
me = 0.51099895000e6
mp = 938.27208816e6
mu = 931.49410242e6
e = 1.602176634e-19
c = 299792458


def cal_tuneshift_KV(N, Z, A, m0, Ek, R, Qx, Qy, sigmax_rms, sigmay_rms):
    gamma = Ek / m0 + 1
    beta = np.sqrt(1 - 1 / gamma / gamma)

    I = N * Z * e / (2 * np.pi * R / (beta * c))
    E = A * gamma * m0 * e

    K = Z * e * I / (2 * np.pi * epsilon0 * E * gamma**2 * beta**3 * c)

    delta_Qx = (
        -1 * R**2 / Qx * K * (1 / (4 * (sigmax_rms * (sigmax_rms + sigmay_rms))))
    )  # For KV distribution, full_emittance = 4*rms_emittance
    delta_Qy = -1 * R**2 / Qy * K * (1 / (4 * (sigmay_rms * (sigmax_rms + sigmay_rms))))

    print(delta_Qx, delta_Qy)
    print(Qx + delta_Qx, Qy + delta_Qy)


def cal_tuneshift_GAUSS(N, Z, A, m0, Ek, R, Qx, Qy, sigmax_rms, sigmay_rms):
    gamma = Ek / m0 + 1
    beta = np.sqrt(1 - 1 / gamma / gamma)

    I = N * Z * e / (2 * np.pi * R / (beta * c))
    E = A * gamma * m0 * e

    K = Z * e * I / (2 * np.pi * epsilon0 * E * gamma**2 * beta**3 * c)

    delta_Qx = (
        -1 * R**2 / Qx * K / 2 * (1 / (1 * (sigmax_rms * (sigmax_rms + sigmay_rms))))
    )
    delta_Qy = (
        -1 * R**2 / Qy * K / 2 * (1 / (1 * (sigmay_rms * (sigmax_rms + sigmay_rms))))
    )

    print(delta_Qx, delta_Qy)
    print(Qx + delta_Qx, Qy + delta_Qy)


def plot_tune_spread(filepath, nux_theory, nuy_theory):
    nux, nuy = np.loadtxt(
        filepath, delimiter=",", usecols=(3, 4), unpack=True, skiprows=1
    )

    fig, ax = plt.subplots()

    ax.scatter(nux, nuy, alpha=0.01, s=0.1, edgecolors=None)
    ax.set_xlabel(r"$\mathrm{\nu_x}$")
    ax.set_ylabel(r"$\mathrm{\nu_y}$")

    ax.scatter(nux_theory, nuy_theory)
    # ax.scatter(0.47, 0.43)
    # ax.set_xlim(0.425, 0.46)
    # ax.set_ylim(0.375, 0.405)
    plt.show()


if __name__ == "__main__":
    cal_tuneshift_KV(
        N=3e11,
        Z=19,
        A=78,
        m0=mu,
        Ek=30e6,
        R=569.1 / (2 * np.pi),
        Qx=9.47,
        Qy=9.43,
        sigmax_rms=np.sqrt(50e-6 * 9.564422187285917),
        sigmay_rms=np.sqrt(25e-6 * 9.604992376839624),
    )

    plot_tune_spread(
        r"D:\PassSimulation\tuneSpread\2025_0826\1505_39\1505_39_phase_beam0_78Kr19+_bunch0_turn_1_100.csv",
        nux_theory=0.440405665915991,
        nuy_theory=0.388058620276732,
    )
    plot_tune_spread(
        r"D:\PassSimulation\tuneSpread\2025_0826\1539_03\1539_03_phase_beam0_78Kr19+_bunch0_turn_1_100.csv",
        nux_theory=0.440405665915991,
        nuy_theory=0.388058620276732,
    )
    plot_tune_spread(
        r"D:\PassSimulation\tuneSpread\2025_0826\1544_55\1544_55_phase_beam0_78Kr19+_bunch0_turn_1_100.csv",
        nux_theory=0.440405665915991,
        nuy_theory=0.388058620276732,
    )
    plot_tune_spread(
        r"D:\PassSimulation\tuneSpread\2025_0826\1609_13\1609_13_phase_beam0_78Kr19+_bunch0_turn_1_100.csv",
        nux_theory=0.440405665915991,
        nuy_theory=0.388058620276732,
    )
