"""Generate smooth-approximation twiss points.

When no MADX lattice is available, a simple smooth ring with constant
beta functions can be used. Beta = C / (2πQ).
"""

import numpy as np

from PASS.para.schema.twiss import TwissPoint


def generate_smooth_twiss(
    circumference: float,
    qx: float,
    qy: float,
    num_points: int,
    alpha_x: float = 0.0,
    alpha_y: float = 0.0,
    dx: float = 0.0,
    dpx: float = 0.0,
    muz: float = 0.0,
    dqx: float = 0.0,
    dqy: float = 0.0,
    longitudinal_transfer: str = "off",
) -> tuple[list[TwissPoint], float]:
    """Generate smooth-approximation twiss points.

    Beta functions are constant: βx = C/(2πQx), βy = C/(2πQy).
    Phase advances linearly from 0 to Q.

    Args:
        circumference: ring circumference in meters.
        qx, qy: horizontal/vertical tune.
        num_points: number of twiss points.
        alpha_x, alpha_y: alpha functions (constant, default 0).
        dx, dpx: dispersion (constant, default 0).
        muz: longitudinal tune.
        dqx, dqy: chromaticity.
        longitudinal_transfer: "off" / "drift" / "matrix".

    Returns:
        (list of TwissPoint, circumference)
    """
    betax = circumference / (2.0 * np.pi * qx)
    betay = circumference / (2.0 * np.pi * qy)

    print(f"[Smooth Twiss] C={circumference}, βx={betax:.4f}, βy={betay:.4f}, "
          f"Qx={qx}, Qy={qy}, DQx={dqx}, DQy={dqy}")

    s = np.linspace(0, circumference, num_points, endpoint=True)
    mux = np.linspace(0, qx, num_points, endpoint=True)
    muy = np.linspace(0, qy, num_points, endpoint=True)

    items = []
    for i in range(num_points):
        if i == 0:
            tp = TwissPoint(
                s=s[i], s_previous=s[i],
                alpha_x=alpha_x, alpha_y=alpha_y,
                beta_x=betax, beta_y=betay,
                mu_x=mux[i], mu_y=muy[i], mu_z=0.0,
                dx=dx, dpx=dpx,
                alpha_x_previous=alpha_x, alpha_y_previous=alpha_y,
                beta_x_previous=betax, beta_y_previous=betay,
                mu_x_previous=mux[i], mu_y_previous=muy[i], mu_z_previous=0.0,
                dx_previous=dx, dpx_previous=dpx,
                dqx=0.0, dqy=0.0,
                longitudinal_transfer=longitudinal_transfer,
            )
        else:
            tp = TwissPoint(
                s=s[i], s_previous=s[i - 1],
                alpha_x=alpha_x, alpha_y=alpha_y,
                beta_x=betax, beta_y=betay,
                mu_x=mux[i], mu_y=muy[i],
                mu_z=s[i] / circumference * muz,
                dx=dx, dpx=dpx,
                alpha_x_previous=alpha_x, alpha_y_previous=alpha_y,
                beta_x_previous=betax, beta_y_previous=betay,
                mu_x_previous=mux[i - 1], mu_y_previous=muy[i - 1],
                mu_z_previous=s[i - 1] / circumference * muz,
                dx_previous=dx, dpx_previous=dpx,
                dqx=dqx * (mux[i] - mux[i - 1]) / qx,
                dqy=dqy * (muy[i] - muy[i - 1]) / qy,
                longitudinal_transfer=longitudinal_transfer,
            )
        items.append(tp)

    print(f"[Smooth Twiss] {len(items)} smooth twiss points generated")
    return items, circumference
