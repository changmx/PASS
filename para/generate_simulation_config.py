import numpy as np
import os
import json


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
        "Output directory": "D:\PassSimulation",
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
                "Number of macro particles per bunch": 1e6,
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
                "Transverse dist": "kv",  # [kv/gaussian/uniform]
                "Logitudinal dist": "gaussian", # [gaussian/uniform]
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
        Sequence.update(Transfer)

        Spacecharge = {
            "Space charge" + str(i): {"S (m)": 0 + i * 10},
            "Command": "Spacecharge",
        }
        # for key in Spacecharge_sim_para:
        #     Spacecharge["Space charge" + str(i)][key] = Spacecharge_sim_para[key]

        Sequence.update(Spacecharge)

    Sequencepara = {"Sequence": Sequence}

    ########## config finish ##########

    merged_dict = {**Beampara, **Spacecharge_sim_para, **BeambeamPara, **Sequencepara}

    with open(config_path, "w") as jsonfile:
        json.dump(merged_dict, jsonfile, indent=4)


def generate_linear_lattice_config_beam1(fileName="beam1.json"):

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
        "Output directory": "D:\PassSimulation",
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
                "Number of macro particles per bunch": 1e6,
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
                "Transverse dist": "kv",  # [kv/gaussian/uniform]
                "Logitudinal dist": "gaussian", # [gaussian/uniform]
                "Offset x": {
                    "Is offset": False,
                    "Offset (m)": 0,
                },
                "Offset y": {
                    "Is offset": False,
                    "Offset (m)": 0,
                },
                "Is load distribution": False,
                "Name of loaded file": "name",  # file must be put in "Output directory/distribution/fixed/"
                "Is save initial distribution": True,
            },
        }
    }
    Sequence.update(Injection)

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
        Sequence.update(Transfer)

        Spacecharge = {
            "Space charge" + str(i): {"S (m)": 0 + i * 10},
            "Command": "Spacecharge",
        }
        # for key in Spacecharge_sim_para:
        #     Spacecharge["Space charge" + str(i)][key] = Spacecharge_sim_para[key]

        Sequence.update(Spacecharge)

    Sequencepara = {"Sequence": Sequence}

    ########## config finish ##########

    merged_dict = {**Beampara, **Spacecharge_sim_para, **BeambeamPara, **Sequencepara}

    with open(config_path, "w") as jsonfile:
        json.dump(merged_dict, jsonfile, indent=4)


if __name__ == "__main__":
    generate_linear_lattice_config_beam0()
    generate_linear_lattice_config_beam1()
