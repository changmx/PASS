import numpy as np
import sys


def generate_twiss_smooth_approximate(
    circum,
    mux,
    muy,
    numPoints,
    logi_transfer: str = "drift/matrix/off",
    alfx=0,
    alfy=0,
    Dx=0,
    Dpx=0,
    DQx=0,
    DQy=0,
    muz=0,
):
    betax = circum / (2 * np.pi * mux)
    betay = circum / (2 * np.pi * muy)

    print(f"Circumference (m): {circum}")
    print(f"Betax = {betax}, Betay = {betay}")
    print(f"Mux = {mux}, Muy = {muy}")
    print(f"DQx = {DQx}, DQy = {DQy}")

    s_list = [circum / numPoints * i for i in range(numPoints + 1)]
    mux_list = [mux / numPoints * i for i in range(numPoints + 1)]
    muy_list = [muy / numPoints * i for i in range(numPoints + 1)]

    Lattice_json = []

    for i in np.arange(numPoints + 1):
        if i == 0:
            lattice = {
                f"TwissSmooApprox_{s_list[i]}[{i}]": {
                    "S (m)": s_list[i],
                    "Command": "Twiss",
                    "S previous (m)": s_list[i],
                    "Alpha x": alfx,
                    "Alpha y": alfx,
                    "Beta x (m)": betax,
                    "Beta y (m)": betay,
                    "Mu x": mux_list[i],
                    "Mu y": muy_list[i],
                    "Mu z": 0.0,
                    "Dx (m)": Dx,
                    "Dpx": Dpx,
                    "Alpha x previous": alfx,
                    "Alpha y previous": alfy,
                    "Beta x previous (m)": betax,
                    "Beta y previous (m)": betay,
                    "Mu x previous": mux_list[i],
                    "Mu y previous": muy_list[i],
                    "Mu z previous": 0.0,
                    "Dx (m) previous": Dx,
                    "Dpx previous": Dpx,
                    "DQx": 0.0,
                    "DQy": 0.0,
                    "Longitudinal transfer": logi_transfer,
                },
            }
        else:
            lattice = {
                f"TwissSmooApprox_{s_list[i]}[{i}]": {
                    "S (m)": s_list[i],
                    "Command": "Twiss",
                    "S previous (m)": s_list[i - 1],
                    "Alpha x": alfx,
                    "Alpha y": alfy,
                    "Beta x (m)": betax,
                    "Beta y (m)": betay,
                    "Mu x": mux_list[i],
                    "Mu y": muy_list[i],
                    "Mu z": s_list[i] / circum * (muz - 0),
                    "Dx (m)": Dx,
                    "Dpx": Dpx,
                    "Alpha x previous": alfx,
                    "Alpha y previous": alfy,
                    "Beta x previous (m)": betax,
                    "Beta y previous (m)": betay,
                    "Mu x previous": mux_list[i - 1],
                    "Mu y previous": muy_list[i - 1],
                    "Mu z previous": s_list[i - 1] / circum * (muz - 0),
                    "Dx (m) previous": Dx,
                    "Dpx previous": Dpx,
                    "DQx": DQx * (mux_list[i] - mux_list[i - 1]) / mux,
                    "DQy": DQy * (muy_list[i] - muy_list[i - 1]) / muy,
                    "Longitudinal transfer": logi_transfer,
                },
            }
        # print(name[i] + "_" + str(s[i]))
        Lattice_json.append(lattice)

    return Lattice_json, circum


if __name__ == "__main__":

    pass
