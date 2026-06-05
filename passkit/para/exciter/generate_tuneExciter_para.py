import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd


def generate_tuneExciter_para(
    rf_file, tuneExciter_freq_file, tuneExciter_voltage_file, QKHV_time, num_harmonic
):
    """Generate tune exciter data required by PASS

    Args:
        rf_file (string): cv data
        tuneExciter_freq_file (string): cv data
        tuneExciter_voltage_file (string): cv data
        QKHV_time (int): QKHV trigger time, in unit of ns
        num_harmonic (int): harmonic number of rf frequency
    """

    t_revo, revo = np.loadtxt(
        rf_file, skiprows=0, delimiter=",", unpack=True, usecols=(0, 1)
    )
    t_te, freq = np.loadtxt(
        tuneExciter_freq_file, skiprows=0, delimiter=",", unpack=True, usecols=(0, 1)
    )
    t_te, volt = np.loadtxt(
        tuneExciter_voltage_file, skiprows=0, delimiter=",", unpack=True, usecols=(0, 1)
    )
    freq_times_t = freq * t_te / 1e9

    revo = revo / num_harmonic  # in unit of Hz
    t_te = t_te + QKHV_time

    t = 0
    turn = []
    freq_interp = []
    volt_interp = []

    for i in range(5000):

        if t < t_te[0] or t > t_te[-1]:
            freq_interp.append(0.0)
            volt_interp.append(0.0)
        else:
            pos1 = np.where(t_te > t)[0][0]

            freq_interp.append(freq_times_t[pos1 - 1])
            volt_interp.append(volt[pos1 - 1]/10)

        turn.append(i)

        pos2 = np.where(t_revo > t)[0][0]
        k2 = (revo[pos2] - revo[pos2 - 1]) / (t_revo[pos2] - t_revo[pos2 - 1])
        b2 = revo[pos2] - k2 * (t_revo[pos2])
        revo_interp = k2 * t + b2
        period = int(1 / revo_interp * 1e9)
        t += period

    current_dir = Path(__file__).parent
    savepath = os.path.join(current_dir, "tuneExciter_data.csv")
    savedata = {"turn": turn, "frequency (Hz)": freq_interp, "voltage (V)": volt_interp}
    df_savedata = pd.DataFrame(savedata)
    df_savedata.to_csv(savepath, header=True, index=False, sep=",")

    print(f"Tune exciter data have been saved to file {savepath}")


if __name__ == "__main__":
    generate_tuneExciter_para(
        rf_file=r"D:\PASS\para\exciter\HIAF_BR_RF43LLRF01_Frequency1_w_152046.csv",
        tuneExciter_freq_file=r"D:\PASS\para\exciter\HIAF_BR_BD42TUNE_Frequency1_w_4000.csv",
        tuneExciter_voltage_file=r"D:\PASS\para\exciter\HIAF_BR_BD42TUNE_Voltage_w_4000.csv",
        QKHV_time=10000000,
        num_harmonic=4,
    )
