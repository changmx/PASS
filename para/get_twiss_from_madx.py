import numpy as np
import sys
import os

from cpymad.madx import Madx
from get_error_from_madx import get_error


def gen_twiss_from_madx(
    seq_file,
    error_file=None,
    seq_name="ring",
    logi_transfer_method="off",
    muz=0,
    DQx=0,
    DQy=0,
    centre=False,
    is_beam_in_seq=False,
    particle="proton",
    energy=1000,
    is_add_sextupole=False,
):
    """
    Read madx sequence file and generate twiss parameters.
    If input error file, generate thin multipole elements with field error.

    Args:
        seq_file (str):
            - abspath of sequence file.
        error_file (str, optional. Defaults to None.):
            - abspath of error file. If it is not None, the error data will be read.
        seq_name (str, optional. Defaults to 'ring'.):
            - sequence name to be used in sequence file.
        logi_transfer_method (str: 'drift/matrix/off', optional. Defaults to 'off'.):
            - 'drift': particles are transfered longitudinally by the drift method
            - 'matrix': particles are transfered longitudinally by linear matrix (required muz input)
            - 'off': no longitudinally transfer.
        muz (float, optional. Defaults to 0):
            - only takes effect in the 'matrix' mode, longitudinal tune.
        DQx (float, optional. Defaults to 0):
            - Hor. chromaticity. It is allocated to each transmission point according to the proportion of phase advance.
        DQy (float, optional. Defaults to 0):
            - Ver. chromaticity. It is allocated to each transmission point according to the proportion of phase advance.
        centre (bool, optional. Defaults to false):
            - calculation of the linear lattice functions at the center of the element instead of the end of the element.
        is_beam_in_seq (bool, optional. Defaults to false):
            - Whether there is beam data in sequence file.
        particle (str, optional. Defaults to 'proton'):
            - Beam type for madx beam command.
        energy (float, optional. Defaults to 1000 GeV):
            - Beam energy for madx beam command.
        is_add_sextupole (bool, optional. Defaults to false):
            - Wheter extract sextupole element from twiss data and add it to tracking. Mainly used as thin lens.
    """

    # ------------------ Read sequence and generate twiss ------------------ #

    madx = Madx()
    madx.option(echo=False)

    madx.call(file=seq_file)
    if not is_beam_in_seq:
        madx.command.beam(sequence=seq_name, particle=particle, energy=energy)

    madx.use(sequence=seq_name)

    dir_path = os.path.dirname(seq_file)
    full_name = os.path.basename(seq_file)
    file_name, file_ext = os.path.splitext(full_name)
    output_twiss = os.sep.join([dir_path, file_name + "_twiss.dat"])

    twiss = madx.twiss(sequence=seq_name, file=output_twiss, centre=centre)

    # twiss_pd = twiss.dframe()
    # twiss_pd.to_csv(r"D:\PASS\para\twiss_pd.csv", index=False)

    print(f"Sequence file has been successfully read: {seq_file}")
    print(f"Twiss data has been writed to: {output_twiss}")

    # ------------------ Generate PASS required twiss data ------------------ #

    s = twiss.s
    name = twiss.name
    betx = twiss.betx
    bety = twiss.bety
    alfx = twiss.alfx
    alfy = twiss.alfy
    Dx = twiss.dx
    Dpx = twiss.dpx
    mux = twiss.mux
    muy = twiss.muy
    keyword = twiss.keyword
    k2l = twiss.k2l
    k2sl = twiss.k2sl
    l = twiss.l

    # print(len(twiss.keyword), twiss.keyword)
    # print(len(twiss.k2l), twiss.k2l)

    if s[0] != 0:
        print("The start position of twiss data must be 0, but now is {}".format(s[0]))
        sys.exit()

    circumference = s[-1]
    print("Circumference (m): {}".format(circumference))

    mux_ring = mux[-1]
    muy_ring = muy[-1]
    print(f"Mux = {mux_ring}, Muy = {muy_ring}")
    print(f"DQx = {DQx}, DQy = {DQy}")

    Lattice_json = []
    sextupole_json = []

    for i in np.arange(len(s)):
        if i == 0:
            lattice = {
                name[i]
                + "_"
                + str(s[i]): {
                    "S (m)": s[i],
                    "Command": "Twiss",
                    "S previous (m)": s[i],
                    "Alpha x": alfx[i],
                    "Alpha y": alfy[i],
                    "Beta x (m)": betx[i],
                    "Beta y (m)": bety[i],
                    "Mu x": mux[i],
                    "Mu y": muy[i],
                    "Mu z": 0.0,
                    "Dx (m)": Dx[i],
                    "Dpx": Dpx[i],
                    "Alpha x previous": alfx[i],
                    "Alpha y previous": alfy[i],
                    "Beta x previous (m)": betx[i],
                    "Beta y previous (m)": bety[i],
                    "Mu x previous": mux[i],
                    "Mu y previous": muy[i],
                    "Mu z previous": 0.0,
                    "Dx (m) previous": Dx[i],
                    "Dpx previous": Dpx[i],
                    "DQx": 0.0,
                    "DQy": 0.0,
                    "Longitudinal transfer": logi_transfer_method,
                },
            }
            if is_add_sextupole:
                if keyword[i] == "sextupole":
                    if np.abs(k2l[i]) > 1e-10 and np.abs(k2sl[i]) < 1e-10:
                        sext_dict = {
                            name[i]
                            + "_elem_"
                            + str(s[i]): {
                                "S (m)": s[i],
                                "Command": "SextupoleNormElement",
                                "L (m)": l[i],
                                "Drift length (m)": 0,
                                "k2 (m^-3)": k2l[i] / l[i],
                                "isFieldError": False,
                                "Error order": 0,
                                "KNL": [],
                                "KSL": [],
                                "Is thin lens": True,
                            }
                        }
                    elif np.abs(k2l[i]) < 1e-10 and np.abs(k2sl[i]) > 1e-10:
                        sext_dict = {
                            name[i]
                            + "_elem_"
                            + str(s[i]): {
                                "S (m)": s[i],
                                "Command": "SextupoleSkewElement",
                                "L (m)": l[i],
                                "Drift length (m)": 0,
                                "k2s (m^-3)": k2sl[i] / l[i],
                                "isFieldError": False,
                                "Error order": 0,
                                "KNL": [],
                                "KSL": [],
                                "Is thin lens": True,
                            },
                        }
                    elif np.abs(k2l[i]) < 1e-10 and np.abs(k2sl[i]) < 1e-10:
                        sext_dict = {
                            name[i]
                            + "_elem_"
                            + str(s[i]): {
                                "S (m)": s[i],
                                "Command": "SextupoleNormElement",
                                "L (m)": l[i],
                                "Drift length (m)": 0,
                                "k2 (m^-3)": k2l[i] / l[i],
                                "isFieldError": False,
                                "Error order": 0,
                                "KNL": [],
                                "KSL": [],
                                "Is thin lens": True,
                            }
                        }
                    else:
                        print(
                            f"Error: Sextupole: k2 = {k2l[i]/l[i]}, k2s = {k2sl[i]/l[i]}, there should be and only 1 variable equal to 0"
                        )
                        sys.exit(1)

        else:
            lattice = {
                name[i]
                + "_"
                + str(s[i]): {
                    "S (m)": s[i],
                    "Command": "Twiss",
                    "S previous (m)": s[i - 1],
                    "Alpha x": alfx[i],
                    "Alpha y": alfy[i],
                    "Beta x (m)": betx[i],
                    "Beta y (m)": bety[i],
                    "Mu x": mux[i],
                    "Mu y": muy[i],
                    "Mu z": s[i] / circumference * (muz - 0),
                    "Dx (m)": Dx[i],
                    "Dpx": Dpx[i],
                    "Alpha x previous": alfx[i - 1],
                    "Alpha y previous": alfy[i - 1],
                    "Beta x previous (m)": betx[i - 1],
                    "Beta y previous (m)": bety[i - 1],
                    "Mu x previous": mux[i - 1],
                    "Mu y previous": muy[i - 1],
                    "Mu z previous": s[i - 1] / circumference * (muz - 0),
                    "Dx (m) previous": Dx[i - 1],
                    "Dpx previous": Dpx[i - 1],
                    "DQx": DQx * (mux[i] - mux[i - 1]) / mux_ring,
                    "DQy": DQy * (muy[i] - muy[i - 1]) / muy_ring,
                    "Longitudinal transfer": logi_transfer_method,
                },
            }
            if is_add_sextupole:
                if keyword[i] == "sextupole":
                    if np.abs(k2l[i]) > 1e-10 and np.abs(k2sl[i]) < 1e-10:
                        sext_dict = {
                            name[i]
                            + "_elem_"
                            + str(s[i]): {
                                "S (m)": s[i],
                                "Command": "SextupoleNormElement",
                                "L (m)": l[i],
                                "Drift length (m)": 0,
                                "k2 (m^-3)": k2l[i] / l[i],
                                "isFieldError": False,
                                "Error order": 0,
                                "KNL": [],
                                "KSL": [],
                                "Is thin lens": True,
                            }
                        }
                    elif np.abs(k2l[i]) < 1e-10 and np.abs(k2sl[i]) > 1e-10:
                        sext_dict = {
                            name[i]
                            + "_elem_"
                            + str(s[i]): {
                                "S (m)": s[i],
                                "Command": "SextupoleSkewElement",
                                "L (m)": l[i],
                                "Drift length (m)": 0,
                                "k2s (m^-3)": k2sl[i] / l[i],
                                "isFieldError": False,
                                "Error order": 0,
                                "KNL": [],
                                "KSL": [],
                                "Is thin lens": True,
                            }
                        }
                    elif np.abs(k2l[i]) < 1e-10 and np.abs(k2sl[i]) < 1e-10:
                        sext_dict = {
                            name[i]
                            + "_elem_"
                            + str(s[i]): {
                                "S (m)": s[i],
                                "Command": "SextupoleNormElement",
                                "L (m)": l[i],
                                "Drift length (m)": 0,
                                "k2 (m^-3)": k2l[i] / l[i],
                                "isFieldError": False,
                                "Error order": 0,
                                "KNL": [],
                                "KSL": [],
                                "Is thin lens": True,
                            }
                        }
                    else:
                        print(
                            f"Error: Sextupole: k2 = {k2l[i]/l[i]}, k2s = {k2sl[i]/l[i]}, there should be and only 1 variable equal to 0"
                        )
                        sys.exit(1)

        # print(name[i] + "_" + str(s[i]))
        Lattice_json.append(lattice)
        if is_add_sextupole:
            if keyword[i] == "sextupole":
                sextupole_json.append(sext_dict)
    madx.quit()
    print(f"Twiss points: {len(Lattice_json)}")
    print(f"Sextupole points: {len(sextupole_json)}")

    # ------------------ Generate PASS required error data ------------------ #

    error_json = []

    if error_file != None:
        if not centre:
            print(f"Error: When introducing errors, centre must be true")
            sys.exit(1)

        error_dict = get_error(
            seq_file, error_file, seq_name, is_beam_in_seq, particle, energy
        )
        print(f"Field error data has beed read to dict, size = {len(error_dict)}")

        for key, sub_dict in error_dict.items():
            idx = np.where(name == key)
            s_error = s[idx][0]
            # print(
            #     f"s(twiss) = {s_error}, s(error) = {sub_dict["s"]}, diff = {s_error-sub_dict["s"]}"
            # )
            thin_error_multipole = {
                key
                + "_error_"
                + str(s_error): {
                    "S (m)": s_error,
                    "Command": "MultipoleElement",
                    "L (m)": sub_dict["length"],
                    "Drift length (m)": 0,
                    "Error order": sub_dict["errorOrder"],
                    "KNL": sub_dict["knl"],
                    "KSL": sub_dict["ksl"],
                    "Is thin lens": True,
                }
            }
            error_json.append(thin_error_multipole)

    print(f"Number of error points: {len(error_json)}")

    # ------------------------------ Finished ------------------------------ #

    return Lattice_json + sextupole_json + error_json, circumference


if __name__ == "__main__":
    gen_twiss_from_madx(
        seq_file=r"D:\PASS\para\BRING2021_03_02.seq",
        error_file=r"D:\PASS\para\error.madx",
        centre=True,
    )
