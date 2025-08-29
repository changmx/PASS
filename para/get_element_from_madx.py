import numpy as np
import sys

from cpymad.madx import Madx
from get_error_from_madx import get_error
from get_error_from_madx import convert_string

class_map = {
    "marker": "MarkerElement",
    "sbend": "SBendElement",
    "rbend": "RBendElement",
    "quadrupole": "QuadrupoleElement",
    "sextupole norm": "SextupoleNormElement",
    "sextupole skew": "SextupoleSkewElement",
    "octupole": "OctupoleElement",
    "hkicker": "HKickerElement",
    "vkicker": "VKickerElement",
    # "multipole": "MultipoleElement",
}


def get_element_from_madx(
    seq_file,
    error_file=None,
    seq_name="ring",
    is_beam_in_seq=False,
    particle="proton",
    energy=1000,
):
    """
    Read madx sequence file and generate element list.
    If input error file, add field error to corresponding element.

    Args:
        seq_file (str):
            - abspath of sequence file.
        error_file (str, optional. Defaults to None.):
            - abspath of error file. If it is not None, the error data will be read.
        seq_name (str, optional. Defaults to 'ring'.):
            - sequence name to be used in sequence file.
        is_beam_in_seq (bool, optional. Defaults to false):
            - Whether there is beam data in sequence file.
        particle (str, optional. Defaults to 'proton'):
            - Beam type for madx beam command.
        energy (float, optional. Defaults to 1000 GeV):
            - Beam energy for madx beam command.
    """

    # ------------------------- Read madx sequence ------------------------- #

    madx = Madx()
    madx.option(echo=False)

    madx.call(file=seq_file)
    if not is_beam_in_seq:
        madx.command.beam(sequence=seq_name, particle=particle, energy=energy)
    madx.use(sequence=seq_name)

    seq = madx.sequence[seq_name].elements
    num_elem = len(seq)
    # print(madx.sequence[seq_name])
    # print(madx.sequence[seq_name].elements)
    print(f"Size of sequence file '{seq_file}' = {num_elem}")
    # print(seq["rb"])

    # ------------------ Generate PASS required element data ---------------- #

    elem_dict = {}

    for i in range(num_elem):
        # print(seq[i]._attr)   # show all elements attributes
        node_name = seq[i].node_name
        name = convert_string(node_name)
        if name in elem_dict:
            print(f"Error: element '{name}' is already exists in element dict")
            sys.exit(1)

        s = seq[i].at
        l = seq[i].length
        elem_type = seq[i].base_name

        if elem_type == "marker":
            elem_dict[name] = {
                "S (m)": s,
                "Command": class_map["marker"],
                "L (m)": l,
                "Drift length (m)": 0,
            }
        elif elem_type == "sbend":
            fint = seq[i].fint
            fintx = seq[i].fintx
            if fintx <= 0:
                fintx = fint

            elem_dict[name] = {
                "S (m)": s,
                "Command": class_map["sbend"],
                "L (m)": l,
                "Drift length (m)": 0,
                "angle (rad)": seq[i].angle,
                "e1 (rad)": seq[i].e1,
                "e2 (rad)": seq[i].e2,
                "hgap (m)": seq[i].hgap,
                "fint": fint,
                "fintx": fintx,
                "isFieldError": False,
                "Error order": 0,
                "KNL": [],
                "KSL": [],
            }
        elif elem_type == "rbend":
            fint = seq[i].fint
            fintx = seq[i].fintx
            if fintx <= 0:
                fintx = fint

            elem_dict[name] = {
                "S (m)": s,
                "Command": class_map["rbend"],
                "L (m)": l,
                "Drift length (m)": 0,
                "angle (rad)": seq[i].angle,
                "e1 (rad)": seq[i].e1,
                "e2 (rad)": seq[i].e2,
                "hgap (m)": seq[i].hgap,
                "fint": fint,
                "fintx": fintx,
                "isFieldError": False,
                "Error order": 0,
                "KNL": [],
                "KSL": [],
            }
        elif elem_type == "quadrupole":
            elem_dict[name] = {
                "S (m)": s,
                "Command": class_map["quadrupole"],
                "L (m)": l,
                "Drift length (m)": 0,
                "k1 (m^-2)": seq[i].k1,
                "k1s (m^-2)": seq[i].k1s,
                "isFieldError": False,
                "Error order": 0,
                "KNL": [],
                "KSL": [],
            }
        elif elem_type == "sextupole":
            if np.abs(seq[i].k2) > 1e-10 and np.abs(seq[i].k2s) < 1e-10:
                elem_dict[name] = {
                    "S (m)": s,
                    "Command": class_map["sextupole norm"],
                    "L (m)": l,
                    "Drift length (m)": 0,
                    "k2 (m^-3)": seq[i].k2,
                    "isFieldError": False,
                    "Error order": 0,
                    "KNL": [],
                    "KSL": [],
                    "Is thin lens": False,
                }
            elif np.abs(seq[i].k2) < 1e-10 and np.abs(seq[i].k2s) > 1e-10:
                elem_dict[name] = {
                    "S (m)": s,
                    "Command": class_map["sextupole skew"],
                    "L (m)": l,
                    "Drift length (m)": 0,
                    "k2s (m^-3)": seq[i].k2s,
                    "isFieldError": False,
                    "Error order": 0,
                    "KNL": [],
                    "KSL": [],
                    "Is thin lens": False,
                }
            elif np.abs(seq[i].k2) < 1e-10 and np.abs(seq[i].k2s) < 1e-10:
                elem_dict[name] = {
                    "S (m)": s,
                    "Command": class_map["sextupole norm"],
                    "L (m)": l,
                    "Drift length (m)": 0,
                    "k2 (m^-3)": seq[i].k2,
                    "isFieldError": False,
                    "Error order": 0,
                    "KNL": [],
                    "KSL": [],
                    "Is thin lens": False,
                }
            else:
                print(
                    f"Error: Sextupole: k2 = {seq[i].k2}, k2s = {seq[i].k2s}, there should be and only 1 variable equal to 0"
                )
                sys.exit(1)
        elif elem_type == "octupole":
            elem_dict[name] = {
                "S (m)": s,
                "Command": class_map["octupole"],
                "L (m)": l,
                "Drift length (m)": 0,
                "k3 (m^-4)": seq[i].k3,
                "k3s (m^-4)": seq[i].k3s,
                "isFieldError": False,
                "Error order": 0,
                "KNL": [],
                "KSL": [],
                "Is thin lens": False,
            }
        elif elem_type == "hkicker":
            elem_dict[name] = {
                "S (m)": s,
                "Command": class_map["hkicker"],
                "L (m)": l,
                "Drift length (m)": 0,
                "kick": seq[i].kick,
                "isFieldError": False,
                "Error order": 0,
                "KNL": [],
                "KSL": [],
                "Is thin lens": False,
            }
        elif elem_type == "vkicker":
            elem_dict[name] = {
                "S (m)": s,
                "Command": class_map["vkicker"],
                "L (m)": l,
                "Drift length (m)": 0,
                "kick": seq[i].kick,
                "isFieldError": False,
                "Error order": 0,
                "KNL": [],
                "KSL": [],
                "Is thin lens": False,
            }
        elif elem_type == "drift":
            pass
        elif elem_type == "monitor":
            pass
        else:
            print(f"Warning: we don't support {elem_type} ({name}) @ S={s} now.")
            continue

    keys = list(elem_dict.keys())
    for i in range(1, len(keys)):
        s = elem_dict[keys[i]]["S (m)"]
        s_previous = elem_dict[keys[i - 1]]["S (m)"]
        l = elem_dict[keys[i]]["L (m)"]
        l_previous = elem_dict[keys[i - 1]]["L (m)"]
        drift_length = s - l / 2 - (s_previous + l_previous / 2)
        elem_dict[keys[i]]["Drift length (m)"] = drift_length

    circumference = seq[-1].at + seq[-1].length / 2

    # ------------------ Generate PASS required error data ------------------ #

    if error_file != None:
        error_dict = get_error(
            seq_file, error_file, seq_name, is_beam_in_seq, particle, energy
        )
        print(f"Error data has beed read to dict, size = {len(error_dict)}")

        error_count = 0
        for key, sub_dict in error_dict.items():
            elem_dict[key]["isFieldError"] = True
            elem_dict[key]["Error order"] = sub_dict["errorOrder"]
            elem_dict[key]["KNL"] = sub_dict["knl"]
            elem_dict[key]["KSL"] = sub_dict["ksl"]
            error_count += 1

        print(f"{error_count} error point has been set to corresponding element")

    madx.quit()

    # ------------------ Generate JSON format data ------------------ #

    element_json = []
    for key, sub_dict in elem_dict.items():
        elem_tmp = {key + "_" + str(sub_dict["S (m)"]): sub_dict}
        # error_json.append(thin_error_multipole)
        element_json.append(elem_tmp)

    print(element_json[0])
    print(element_json[-1])

    return element_json, circumference


if __name__ == "__main__":
    get_element_from_madx(
        seq_file=r"D:\PASS\para\BRING2021_03_02.seq",
        error_file=r"D:\PASS\para\error.madx",
        seq_name="ring",
    )
