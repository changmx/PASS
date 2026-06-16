import numpy as np
import sys


def generate_twiss_smooth_approximate(
    circum,
    Qx,
    Qy,
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
    betax = circum / (2 * np.pi * Qx)
    betay = circum / (2 * np.pi * Qy)

    print(f"[Generate \033[34mSmooth Approx.\033[0m Twiss] Circumference (m): {circum}")
    print(f"[Generate \033[34mSmooth Approx.\033[0m Twiss] Betax = {betax}, Betay = {betay}")
    print(f"[Generate \033[34mSmooth Approx.\033[0m Twiss] Qx = {Qx}, Qy = {Qy}")
    print(f"[Generate \033[34mSmooth Approx.\033[0m Twiss] DQx = {DQx}, DQy = {DQy}")

    s = np.linspace(0, circum, numPoints, endpoint=True)
    mux = np.linspace(0, Qx, numPoints, endpoint=True)
    muy = np.linspace(0, Qy, numPoints, endpoint=True)

    elem_dict = {}

    for i in np.arange(numPoints):
        if i == 0:
            elem_dict[f"TwissSmooApprox[{i+1}]"] = {
                "S (m)": s[i],
                "Command": "Twiss",
                "S previous (m)": s[i],
                "Alpha x": alfx,
                "Alpha y": alfx,
                "Beta x (m)": betax,
                "Beta y (m)": betay,
                "Mu x": mux[i],
                "Mu y": muy[i],
                "Mu z": 0.0,
                "Dx (m)": Dx,
                "Dpx": Dpx,
                "Alpha x previous": alfx,
                "Alpha y previous": alfy,
                "Beta x previous (m)": betax,
                "Beta y previous (m)": betay,
                "Mu x previous": mux[i],
                "Mu y previous": muy[i],
                "Mu z previous": 0.0,
                "Dx (m) previous": Dx,
                "Dpx previous": Dpx,
                "DQx": 0.0,
                "DQy": 0.0,
                "Longitudinal transfer": logi_transfer,
            }

        else:
            elem_dict[f"TwissSmooApprox[{i+1}]"] = {
                "S (m)": s[i],
                "Command": "Twiss",
                "S previous (m)": s[i - 1],
                "Alpha x": alfx,
                "Alpha y": alfy,
                "Beta x (m)": betax,
                "Beta y (m)": betay,
                "Mu x": mux[i],
                "Mu y": muy[i],
                "Mu z": s[i] / circum * (muz - 0),
                "Dx (m)": Dx,
                "Dpx": Dpx,
                "Alpha x previous": alfx,
                "Alpha y previous": alfy,
                "Beta x previous (m)": betax,
                "Beta y previous (m)": betay,
                "Mu x previous": mux[i - 1],
                "Mu y previous": muy[i - 1],
                "Mu z previous": s[i - 1] / circum * (muz - 0),
                "Dx (m) previous": Dx,
                "Dpx previous": Dpx,
                "DQx": DQx * (mux[i] - mux[i - 1]) / Qx,
                "DQy": DQy * (muy[i] - muy[i - 1]) / Qy,
                "Longitudinal transfer": logi_transfer,
            }

    print(f"[Generate \033[34mSmooth Approx.\033[0m Twiss] Success: {len(elem_dict)} smooth approxiamte twiss elements have been generated")

    return elem_dict, circum


if __name__ == "__main__":

    generate_twiss_smooth_approximate(569.1, 9.47, 9.43, 100, "off", DQx=-1, DQy=-1)
