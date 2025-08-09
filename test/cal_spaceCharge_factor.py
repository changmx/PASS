import numpy as np

epsilon0 = 8.8541878128e-12
me = 0.51099895000e6
mp = 938.27208816e6
mu = 931.49410242e6
e = 1.602176634e-19
c = 299792458


def cal_tuneshift_KV(N, Z, A, m0, Ek, R, Qx, Qy, sigmax, sigmay):
    gamma = Ek / m0 + 1
    beta = np.sqrt(1 - 1 / gamma / gamma)

    I = N * Z * e / (2 * np.pi * R / (beta * c))
    E = A * gamma * m0 * e

    K = Z * e * I / (2 * np.pi * epsilon0 * E * gamma**2 * beta**3 * c)

    delta_Qx = -1 * R**2 / Qx * K * (1 / (sigmax * (sigmax + sigmay)))
    delta_Qy = -1 * R**2 / Qy * K * (1 / (sigmay * (sigmax + sigmay)))

    print(delta_Qx, delta_Qy)
    print(Qx + delta_Qx, Qy + delta_Qy)


if __name__ == "__main__":
    cal_tuneshift_KV(
        N=3e11,
        Z=19,
        A=78,
        m0=mu,
        Ek=30e6,
        R=530.7885 / (2 * np.pi),
        Qx=8.45,
        Qy=7.43,
        sigmax=np.sqrt(200e-6 * 10),
        sigmay=np.sqrt(100e-6 * 11.4),
    )
    cal_tuneshift_KV(
        N=3e11,
        Z=19,
        A=78,
        m0=mu,
        Ek=30e6,
        R=569.1 / (2 * np.pi),
        Qx=9.47,
        Qy=9.43,
        sigmax=np.sqrt(200e-6 * 9.564422187285917),
        sigmay=np.sqrt(100e-6 * 9.604992376839624),
    )
