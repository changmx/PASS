import numpy as np
import os
import json
from collections import OrderedDict
from get_twiss_element_from_madx import generate_twiss_json


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
        "Twiss": 1,  # 次级优先级
        "Element": 2,
        "BeamBeam": 3,
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
        "Qx": 0.31,  # <optional>, will not be used if a madx file is loaded. Todo: read ramping file
        "Qy": 0.32,
        "Qz": 0.0125,
        "Chromaticity x": 0,
        "Chromaticity y": 0,
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

    LatticeElement_Sextupole = {
        "LatticeElement_sextupole": {
            "S (m)": 0.2,
            "Command": "Element",
        }
    }
    Sequence.update(LatticeElement_Sextupole)

    Lattice_twiss_list = generate_twiss_json(
        r"D:\AthenaLattice\SZA\v9\sza_sta1.dat", logi_transfer="off"
    )
    for i in Lattice_twiss_list:
        Sequence.update(i)

    LatticeTwiss0 = {
        "LatticeTwiss_oneTurn": {
            "S (m)": 100,
            "Command": "Twiss",
            "Alpha x": 0,
            "Alpha y": 0,
            "Beta x (m)": 0.05,
            "Beta y (m)": 0.1,
            "Mu x": 0.3,
            "Mu y": 0.3,
            # If there is a "Mu z", 6D linear transmission is performed.
            # If there is no "Mu z", a 4D linear transmission is performed,
            # and if there is an additional RF cavity, the RF element will achieve longitudinal motion.
            "Mu z": 0.002,
        },
    }
    # Sequence.update(LatticeTwiss0)

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
                "Is load distribution": False,
                "Name of loaded file": "1600_45_gaussian_proton_bunch0_1000000_superPeriod0.csv",  # file must be put in "Output directory/distribution/fixed/"
                "Is save initial distribution": True,
            },
        }
    }
    Sequence.update(Injection)

    LatticeTwissFile = {
        "LatticeTwissFile": {
            "S (m)": 0,  # If it is the mode of loading twiss file, the value of S is meaningless.
            "Command": "Lattice",
            "Transfer type": "twissfile",  # [twiss/twissfile/element/elementfile]. The transmission method used to transfer from the previous lattice object to the current object
            "File path": "D\\",
        },
    }

    LatticeElementDipole0 = {
        "LatticeElementDipole0": {
            "S (m)": 0,
            "Command": "Lattice",
            "Transfer type": "element",  # [twiss/twissfile/element/elementfile]. The transmission method used to transfer from the previous lattice object to the current object
            "Element type": "Dipole",
            "Length": 1,
            "Angle (rad)": 0.2,
            "E1 (rad)": 0.1,
            "E2 (rad)": 0.1,
            "Fint": 0.5,
        },
    }

    LatticeElementFile = {
        "LatticeElementFile": {
            "S (m)": 0,  # If it is the mode of loading twiss file, the value of S is meaningless.
            "Command": "Lattice",
            "Transfer type": "elementfile",  # [twiss/twissfile/element/elementfile]. The transmission method used to transfer from the previous lattice object to the current object
            "File path": "D\\",
        }
    }

    for i in range(5):
        Transfer = {
            "Transfer"
            + str(i): {
                "S (m)": 0 + i * 10,
                "Command": "Transfer",
                "Alpha x": 0,
                "Alpha y": 0,
                "Beta x (m)": 0.05,
                "Beta y (m)": 0.012,
                "Phase advance x (2pi)": 0.3,
                "Phase advance y (2pi)": 0.3,
            },
        }
        # Sequence.update(Transfer)

        Spacecharge = {
            "SpaceCharge" + str(i): {"S (m)": 0 + i * 10, "Command": "Space charge"}
        }
        # for key in Spacecharge_sim_para:
        #     Spacecharge["Space charge" + str(i)][key] = Spacecharge_sim_para[key]

        # Sequence.update(Spacecharge)

    Sequence = sort_sequence(Sequence)
    Sequencepara = {"Sequence": Sequence}

    ########## config finish ##########

    merged_dict = {**Beampara, **Spacecharge_sim_para, **BeambeamPara, **Sequencepara}

    with open(config_path, "w") as jsonfile:
        json.dump(merged_dict, jsonfile, indent=4)


if __name__ == "__main__":
    generate_linear_lattice_config_beam0()
    # generate_linear_lattice_config_beam1()
