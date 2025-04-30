import numpy as np
import os
import json
from collections import OrderedDict
from get_twiss_from_madx import generate_twiss_json
from get_element_from_madx import generate_element_json


def convert_ordereddict(obj):
    if isinstance(obj, OrderedDict):
        return {k: convert_ordereddict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ordereddict(item) for item in obj]
    else:
        return obj


def sort_sequence(sequence):
    Command_order = {
        "Injection": 0,  # 最高优先级
        "Twiss": 50,  # 次级优先级
        "SBendElement": 100,
        "RBendElement": 100,
        "QuadrupoleElement": 100,
        "SextupoleElement": 100,
        "DistMonitor": 150,
        "BeamBeam": 300,
        "Other": 999,  # 最低优先级
    }

    # 首先按照位置S进行从小到大的排序，如果两个字典的S相同，按照上述Commnad自定义顺序进行排序
    sorted_sequence = OrderedDict(
        sorted(
            sequence.items(),
            key=lambda item: (
                item[1]["S (m)"],  # 主排序键
                Command_order.get(item[1]["Command"], 999),  # 次排序键
            ),
        )
    )

    sorted_sequence_dictType = convert_ordereddict(
        sorted_sequence
    )  # 递归转换，把OrderDict转化为dict类型

    return sorted_sequence_dictType


def generate_linear_lattice_config_beam0(fileName="beam0.json"):

    current_path, _ = os.path.split(__file__)
    parent_path = os.path.dirname(current_path)
    config_path = os.sep.join([parent_path, "para", fileName])
    print("The simulation configuration will be written to file: ", config_path)

    ########## config start ##########

    Beampara = {
        "Name": "proton",  # [particle name]: arbitrary, just to let the user distinguish the beam
        "Number of protons per particle": 1,
        "Number of neutrons per particle": 0,  # if #neutrons is > 0, the mass of the particle is calculated based on nucleon mass
        "Number of charges per particle": 1,
        "Number of bunches per beam": 1,
        "Circumference (m)": 200,
        # "Qx": 0.31,  # <optional>, will not be used if a madx file is loaded. Todo: read ramping file
        # "Qy": 0.32,
        # "Qz": 0.0125,
        # "Chromaticity x": 0,
        # "Chromaticity y": 0,
        "GammaT": 1,
        "Number of turns": 10,
        "Number of GPU devices": 1,
        "Device Id": [0, 3],
        "Output directory": "D:\\PassSimulation",
        "Is plot figure": False,
    }

    # Field solver
    # PIC_conv: using Green function with open boundary condition
    # PIC_FD_dm: in the case of rectangular boundary, solve the matrix after using DST
    # PIC_FD_m: in the case of any boundary, the matrix is solved directly after LU decomposition
    # Eq_quasi_static: using the B-E formula, each calculation is based on the current sigmax, sigmay
    # Eq_frozen: using th B-E formula with unchanged sigmax and sigmay

    Spacecharge_sim_para = {
        "Space charge simulation parameters": {
            "Is space charge": False,
            "Number of grid x": 256,
            "Number of grid y": 256,
            "Grid x length": 1e-5,
            "Grid y length": 3.5e-6,
            "Number of bunch slices": 10,
            "Field solver:": "PIC_conv",  # [PIC_conv/PIC_FD_dm/PIC_FD_m/Eq_quasi_static/Eq_frozen]: field solver
        }
    }

    BeambeamPara = {
        "Beam-beam simulation parameters": {
            "Is beam-beam": False,
            "Number of IP": 1,
            "IP0": {
                "Number of grid x": 256,
                "Number of grid y": 256,
                "Grid x length": 1e-5,
                "Grid y length": 3.5e-6,
                "Number of bunch slices": 10,
                "Field solver:": "PIC_conv",  # [PIC_conv/PIC_FD_dm/PIC_FD_m/Eq_quasi_static/Eq_frozen]: field solver
            },
        }
    }

    Sequence = {}

    # Initialize all bunches' distribution of the beam
    Injection = {
        "Injection": {
            "S (m)": 0,
            "Command": "Injection",
            "bunch0": {
                "Kinetic energy per particle (eV)": 19.08e9,
                "Number of real particles per bunch": 1.05e11,
                "Number of macro particles per bunch": 1e4,
                "Mode": "1turn1time",  # [1turn1time/1turnxtime/xturnxtime]
                "Inject turns": [0],
                "Alpha x": 0,
                "Alpha y": 0,
                "Beta x (m)": 0.05,
                "Beta y (m)": 0.012,
                "Emittance x (m'rad)": 100e-9,
                "Emittance y (m'rad)": 50e-9,
                "Sigma z (m)": 0.08,
                "DeltaP/P": 1.62e-3,
                "Transverse dist": "gaussian",  # [kv/gaussian/uniform]
                "Logitudinal dist": "uniform",  # [gaussian/uniform]
                "Offset x": {
                    "Is offset": False,
                    "Offset (m)": 0,
                },
                "Offset y": {
                    "Is offset": False,
                    "Offset (m)": 0,
                },
                "Is load distribution": True,
                "Name of loaded file": "1557_00_beam0_proton_bunch0_10000_hor_gaussian_longi_uniform_injection.csv",  # file must be put in "Output directory/distribution/fixed/"
                "Is save initial distribution": True,
            },
        }
    }
    Sequence.update(Injection)

    # Spacecharge = {
    #     "SpaceCharge" + str(i): {"S (m)": 0 + i * 10, "Command": "Space charge"}
    # }
    # for key in Spacecharge_sim_para:
    #     Spacecharge["Space charge" + str(i)][key] = Spacecharge_sim_para[key]

    # Sequence.update(Spacecharge)

    # twiss_list_from_madx = generate_twiss_json(
    #     r"D:\AthenaLattice\SZA\v9\sza_sta1.dat", logi_transfer="off"
    # )
    # for twiss in twiss_list_from_madx:
    #     Sequence.update(twiss)

    # element_list_from_madx = generate_element_json(r"D:\AthenaLattice\SZA\v13\sza.seq")
    # for element in element_list_from_madx:
    #     # print(element)
    #     Sequence.update(element)

    lattice_oneturn_map = {
        "oneturn_map": {
            "S (m)": 0,
            "Command": "Twiss",
            "S previous (m)": 0,
            "Alpha x": 0,
            "Alpha y": 0,
            "Beta x (m)": 0.05,
            "Beta y (m)": 0.012,
            "Mu x": 1,
            "Mu y": 1,
            "Mu z": 1,
            "Alpha x previous": 0,
            "Alpha y previous": 0,
            "Beta x previous (m)": 0.05,
            "Beta y previous (m)": 0.012,
            "Mu x previous": 0,
            "Mu y previous": 0,
            # "Dx (m)": Dx[i],
            # "Dpx": Dpx[i],
            "Logitudinal transfer": "matrix",
        },
    }
    Sequence.update(lattice_oneturn_map)

    Monitor_Dist_oneturn = {
        "DistMonitor_oneturn_0": {"S (m)": 0, "Command": "DistMonitor"},
    }
    Sequence.update(Monitor_Dist_oneturn)
    
    Monitor_Stat_oneturn = {
        "StatMonitor_oneturn_0": {"S (m)": 0, "Command": "StatMonitor"},
    }
    Sequence.update(Monitor_Stat_oneturn)

    Sequence = sort_sequence(Sequence)
    Sequencepara = {"Sequence": Sequence}

    ########## config finish ##########

    merged_dict = {**Beampara, **Spacecharge_sim_para, **BeambeamPara, **Sequencepara}

    with open(config_path, "w") as jsonfile:
        json.dump(merged_dict, jsonfile, indent=4)


if __name__ == "__main__":
    generate_linear_lattice_config_beam0()
    # generate_linear_lattice_config_beam1()
