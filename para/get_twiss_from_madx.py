import numpy as np
import sys


def get_allTwiss(filepath):

    data = []

    with open(file=filepath, mode="r") as f:
        for line in f.readlines():
            if "@" in line:
                pass

            elif "*" in line:
                line_tmp = line.replace("*", "")
                line_tmp = line_tmp.split()
                data.append(line_tmp)

            elif "$" in line:
                if "$START" in line or "$END" in line:
                    line_tmp = line.replace("*", "")
                    line_tmp = line_tmp.replace("$", "_")
                    line_tmp = line_tmp.replace('"', "")
                    line_tmp = line_tmp.split()
                    data.append(line_tmp)
                else:
                    pass

            else:
                line_tmp = line.replace("*", "")
                line_tmp = line_tmp.replace('"', "")
                line_tmp = line_tmp.split()
                data.append(line_tmp)

    # Check and delete duplicate twiss data with the same name and the same position
    # This process is mainly aimed at addressing the issue where multiple MARKERs with the same name repeatedly occur at the same position
    seen = set()
    data_remove_duplication = []
    for sub_list in data:
        tuple_sub = tuple(sub_list)
        if tuple_sub not in seen:
            seen.add(tuple_sub)
            data_remove_duplication.append(sub_list)
        else:
            print("Delete duplication data: {} ...".format(sub_list[0:7]))

    data_array = np.array(data_remove_duplication)
    Nrow = data_array.shape[0]
    Ncol = data_array.shape[1]

    return data_array


def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def get_specificTwiss(filepath, twissName, positionName="S"):
    all_twiss = get_allTwiss(filepath)
    Nrow = all_twiss.shape[0]
    Ncol = all_twiss.shape[1]

    positionId = None
    twissId = None

    paraName = all_twiss[0]

    for i in range(Ncol):
        if paraName[i].lower() == positionName.lower():
            positionId = i
            break
    for i in range(Ncol):
        if paraName[i].lower() == twissName.lower():
            twissId = i
            break

    if positionId == None:
        print(
            "Error, para '{0}' is not in the data file as shown below:\n{1}".format(
                positionName, paraName
            )
        )
        sys.exit(1)
    if twissId == None:
        print(
            "Error, para '{0}' is not in the data file as shown below:\n{1}".format(
                twissName, paraName
            )
        )
        sys.exit(1)

    s = []
    twiss = []

    if is_float(all_twiss[1][twissId]):
        for i in range(1, Nrow):
            # range "1": means the first line holding the twiss name is ignored
            s.append(float(all_twiss[i][positionId]))
            twiss.append(float(all_twiss[i][twissId]))
    else:
        for i in range(1, Nrow):
            s.append(float(all_twiss[i][positionId]))
            twiss.append(all_twiss[i][twissId])

    return np.array(s), np.array(twiss)


def generate_twiss_json(
    filepath,
    logi_transfer: str = "drift/matrix/off",
    muz=0,
    DQx=0,
    DQy=0,
):
    # get S, name, beta, alpha, phase, Dx, Dpx from madx twiss file
    s, name = get_specificTwiss(filepath, twissName="name")
    s, betax = get_specificTwiss(filepath, twissName="betx")
    s, betay = get_specificTwiss(filepath, twissName="bety")
    s, alfx = get_specificTwiss(filepath, twissName="alfx")
    s, alfy = get_specificTwiss(filepath, twissName="alfy")
    s, Dx = get_specificTwiss(filepath, twissName="Dx")
    # s, Dpx = get_specificTwiss(filepath, twissName="Dpx")
    s, mux = get_specificTwiss(filepath, twissName="mux")
    s, muy = get_specificTwiss(filepath, twissName="muy")

    circumference = s[-1]
    print("Circumference (m): {}".format(circumference))

    mux_ring = mux[-1]
    muy_ring = muy[-1]
    print(f"Mux = {mux_ring}, Muy = {muy_ring}")
    print(f"DQx = {DQx}, DQy = {DQy}")

    if s[0] != 0:
        print("The start position of twiss data must be 0, but now is {}".format(s[0]))
        sys.exit()

    Lattice_json = []

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
                    "Beta x (m)": betax[i],
                    "Beta y (m)": betay[i],
                    "Mu x": mux[i],
                    "Mu y": muy[i],
                    "Mu z": 0.0,
                    "Alpha x previous": alfx[i],
                    "Alpha y previous": alfy[i],
                    "Beta x previous (m)": betax[i],
                    "Beta y previous (m)": betay[i],
                    "Mu x previous": mux[i],
                    "Mu y previous": muy[i],
                    "Mu z previous": 0.0,
                    "Dx (m)": Dx[i],
                    "Dx (m) previous": Dx[i],
                    "DQx": 0.0,
                    "DQy": 0.0,
                    "Longitudinal transfer": logi_transfer,
                },
            }
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
                    "Beta x (m)": betax[i],
                    "Beta y (m)": betay[i],
                    "Mu x": mux[i],
                    "Mu y": muy[i],
                    "Mu z": s[i] / circumference * (muz - 0),
                    "Alpha x previous": alfx[i - 1],
                    "Alpha y previous": alfy[i - 1],
                    "Beta x previous (m)": betax[i - 1],
                    "Beta y previous (m)": betay[i - 1],
                    "Mu x previous": mux[i - 1],
                    "Mu y previous": muy[i - 1],
                    "Mu z previous": s[i - 1] / circumference * (muz - 0),
                    "Dx (m)": Dx[i],
                    "Dx (m) previous": Dx[i - 1],
                    "DQx": DQx * (mux[i] - mux[i - 1]) / mux_ring,
                    "DQy": DQy * (muy[i] - muy[i - 1]) / muy_ring,
                    "Longitudinal transfer": logi_transfer,
                },
            }
        # print(name[i] + "_" + str(s[i]))
        Lattice_json.append(lattice)

    return Lattice_json, circumference


if __name__ == "__main__":
    generate_twiss_json(r"D:\AthenaLattice\SZA\v13\ring.dat")
