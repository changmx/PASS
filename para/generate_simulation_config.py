import numpy as np
import os
import json
import sys
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
        "MarkerElement": 100,
        "SBendElement": 100,
        "RBendElement": 100,
        "QuadrupoleElement": 100,
        "SextupoleElement": 100,
        "OctupoleElement": 100,
        "HKickerElement": 100,
        "VKickerElement": 100,
        "RFElement": 100,
        "ElSeparatorElement": 100,
        "SpaceCharge": 120,
        "DistMonitor": 150,
        "StatMonitor": 150,
        "ParticleMonitor": 150,
        "SortBunch": 200,
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


def generate_simulation_config_beam0(fileName="beam0.json"):

    current_path, _ = os.path.split(__file__)
    parent_path = os.path.dirname(current_path)
    config_path = os.sep.join([parent_path, "para", fileName])
    print("The simulation configuration will be written to file: ", config_path)

    ###################################### Config global parameters ######################################

    Beampara = {
        "Name": "proton",  # [particle name]: arbitrary, just to let the user distinguish the beam
        "Number of protons per particle": 1,
        "Number of neutrons per particle": 0,  # if #neutrons is > 0, the mass of the particle is calculated based on nucleon mass
        "Number of charges per particle": 1,
        "Number of bunches per beam": 1,
        "Circumference (m)": 136.6,
        # "Qx": 0.31,  # <optional>, will not be used if a madx file is loaded. Todo: read ramping file
        # "Qy": 0.32,
        # "Qz": 0.0125,
        # "Chromaticity x": 0,
        # "Chromaticity y": 0,
        "GammaT": 1,
        "Number of turns": 5000,
        "Number of GPU devices": 1,
        "Device Id": [0],
        "Output directory": "D:\\PassSimulation",
        "Is plot figure": False,
    }

    circumference = Beampara["Circumference (m)"]

    # Field solver
    # PIC_conv: using Green function with open boundary condition
    # PIC_FD_dm: in the case of rectangular boundary, solve the matrix after using DST
    # PIC_FD_m: in the case of any boundary, the matrix is solved directly after LU decomposition
    # Eq_quasi_static: using the B-E formula, each calculation is based on the current sigmax, sigmay
    # Eq_frozen: using th B-E formula with unchanged sigmax and sigmay
    Spacecharge_sim_para = {
        "Space-charge simulation parameters": {
            "Is enable space charge": False,
            "Number of slices": 10,
            "Slice model": "Equal length",  # [Equal particle/Equal length]
            "Field solver": "PIC_conv",  # [PIC_conv/PIC_FD_dm/PIC_FD_m/Eq_quasi_static/Eq_frozen]
        }
    }

    BeambeamPara = {
        "Beam-beam simulation parameters": {
            "Is enable beam-beam": False,
            "Number of slices": 10,
            "Slice model": "Equal particle",  # [Equal particle/Equal length]
            "Field solver": "PIC_conv",  # [PIC_conv/PIC_FD_dm/PIC_FD_m/Eq_quasi_static/Eq_frozen]
            "IP0": {
                "Number of grid x": 256,
                "Number of grid y": 256,
                "Grid x length": 1e-5,
                "Grid y length": 3.5e-6,
            },
        }
    }

    ParticleMonitorPara = {
        "Particle Monitor parameters": {
            "Is enable particle monitor": True,
            "Number of particles to save": 21,
            "Save turn range": [1, 100000, 1],
            "Observer position S (m)": [0, 26.84],
        }
    }

    ###################################### Start create sequence ######################################

    Sequence = {}

    ###################################### Injection parameters ######################################

    # Initialize all bunches' distribution of the beam
    Injection = {
        "Injection": {
            "S (m)": 0,
            "Command": "Injection",
            "bunch0": {
                "Kinetic energy per nucleon (eV/u)": 2000e9,
                "Number of real particles per bunch": 1e4,
                "Number of macro particles per bunch": 1e4,
                "Mode": "1turn1time",  # [1turn1time/1turnxtime/xturnxtime]
                "Inject turns": [1],
                "Alpha x": 0,
                "Alpha y": 0,
                "Beta x (m)": 10.21803918,
                "Beta y (m)": 1.396696787,
                "Emittance x (m'rad)": 28e-6,
                "Emittance y (m'rad)": 28e-6,
                "Dx (m)": 0.0,
                "Dpx": 0.0,
                "Sigma z (m)": 0.1,
                "DeltaP/P": 5e-3,
                "Transverse dist": "kv",  # [kv/gaussian/uniform]
                "Longitudinal dist": "gaussian",  # [gaussian/uniform]
                "Offset x": {
                    "Is offset": False,
                    "Offset (m)": 0,
                },
                "Offset y": {
                    "Is offset": False,
                    "Offset (m)": 0,
                },
                "Is load distribution": False,
                "Name of loaded file": "1511_49_beam0_proton_bunch0_10000_hor_kv_longi_gaussian_Dx_0.000000_injection.csv",  # file must be put in "Output directory/distribution/fixed/"
                "Is save initial distribution": True,
            },
        }
    }
    Sequence.update(Injection)

    ###################################### Get twiss  from madx ######################################

    twiss_list_from_madx, circumference_twissFile = generate_twiss_json(
        r"D:\AthenaLattice\Ion-Track-etched-Membrane\v9-3\ring.dat",
        logi_transfer="off",
        muz=0.0123,
        DQx=-0,
        DQy=-0,
    )
    if (abs(circumference - circumference_twissFile)) > 1e-9:
        print(
            f"Error: Circumference from Beampara dict is = {circumference}, circumference from madx twiss file is {circumference_twissFile}. Check it!"
        )
        sys.exit(1)
    for twiss in twiss_list_from_madx:
        Sequence.update(twiss)

    ###################################### Space charge according to madx twiss ######################################

    # spacecharge_list = []
    # for ielem in twiss_list_from_madx:
    #     first_key = next(iter(ielem))
    #     name_sc = first_key
    #     s_sc = ielem[first_key]["S (m)"]
    #     s_pre_sc = ielem[first_key]["S previous (m)"]

    #     dict_sc = {
    #         name_sc
    #         + "_sc": {
    #             "S (m)": s_sc,
    #             "Command": "SpaceCharge",
    #             "Length (m)": s_sc - s_pre_sc,
    #         }
    #     }
    #     spacecharge_list.append(dict_sc)

    # for dict_sc in spacecharge_list:
    #     Sequence.update(dict_sc)
    # print(f"Number of space charge points: {len(spacecharge_list)}")

    ###################################### Input element ######################################

    SF1_ARC1_1 = {
        "sf1_arc1_1": {
            "S (m)": 8.37,
            "Command": "SextupoleNormElement",
            "L (m)": 0.2,
            "Drift length (m)": 0,
            "k2 (m^-3)": 2.768917952,
            "isFieldError": False,
            "isIgnoreLength": True,
        },
    }
    SF1_ARC1_2 = {
        "sf1_arc1_2": {
            "S (m)": 59.93,
            "Command": "SextupoleNormElement",
            "L (m)": 0.2,
            "Drift length (m)": 0,
            "k2 (m^-3)": 2.768917952,
            "isFieldError": False,
            "isIgnoreLength": True,
        },
    }
    SF1_ARC1_3 = {
        "sf1_arc1_3": {
            "S (m)": 76.67,
            "Command": "SextupoleNormElement",
            "L (m)": 0.2,
            "Drift length (m)": 0,
            "k2 (m^-3)": 2.768917952,
            "isFieldError": False,
            "isIgnoreLength": True,
        },
    }
    SF1_ARC1_4 = {
        "sf1_arc1_4": {
            "S (m)": 128.23,
            "Command": "SextupoleNormElement",
            "L (m)": 0.2,
            "Drift length (m)": 0,
            "k2 (m^-3)": 2.768917952,
            "isFieldError": False,
            "isIgnoreLength": True,
        },
    }

    SD1_ARC1_1 = {
        "sd1_arc1_1": {
            "S (m)": 10.82,
            "Command": "SextupoleNormElement",
            "L (m)": 0.2,
            "Drift length (m)": 0,
            "k2 (m^-3)": -3.493098505,
            "isFieldError": False,
            "isIgnoreLength": True,
        },
    }
    SD1_ARC1_2 = {
        "sd1_arc1_2": {
            "S (m)": 57.48,
            "Command": "SextupoleNormElement",
            "L (m)": 0.2,
            "Drift length (m)": 0,
            "k2 (m^-3)": -3.493098505,
            "isFieldError": False,
            "isIgnoreLength": True,
        },
    }
    SD1_ARC1_3 = {
        "sd1_arc1_3": {
            "S (m)": 79.12,
            "Command": "SextupoleNormElement",
            "L (m)": 0.2,
            "Drift length (m)": 0,
            "k2 (m^-3)": -3.493098505,
            "isFieldError": False,
            "isIgnoreLength": True,
        },
    }
    SD1_ARC1_4 = {
        "sd1_arc1_4": {
            "S (m)": 125.78,
            "Command": "SextupoleNormElement",
            "L (m)": 0.2,
            "Drift length (m)": 0,
            "k2 (m^-3)": -3.493098505,
            "isFieldError": False,
            "isIgnoreLength": True,
        },
    }

    SF1_STA1_1 = {
        "sf1_sta1_1": {
            "S (m)": 31.55,
            "Command": "SextupoleNormElement",
            "L (m)": 0.2,
            "Drift length (m)": 0,
            "k2 (m^-3)": 3,
            "isFieldError": False,
            "isIgnoreLength": True,
            "isRamping": True,
            "Ramping file path": r"D:\PASS\para\sf1_sta1_ramping.csv",
        },
    }
    SF1_STA1_2 = {
        "sf1_sta1_2": {
            "S (m)": 41.96,
            "Command": "SextupoleNormElement",
            "L (m)": 0.2,
            "Drift length (m)": 0,
            "k2 (m^-3)": 3,
            "isFieldError": False,
            "isIgnoreLength": True,
            "isRamping": True,
            "Ramping file path": r"D:\PASS\para\sf1_sta1_ramping.csv",
        },
    }

    ElSeparatorElement_entrance = {
        "ES_26.84": {
            "Command": "ElSeparatorElement",
            "S (m)": 26.84,
            "ES Horizontal position (m)": 0.04,
        }
    }

    # Sequence.update(SF1_ARC1_1)
    # Sequence.update(SF1_ARC1_2)
    # Sequence.update(SF1_ARC1_3)
    # Sequence.update(SF1_ARC1_4)

    # Sequence.update(SD1_ARC1_1)
    # Sequence.update(SD1_ARC1_2)
    # Sequence.update(SD1_ARC1_3)
    # Sequence.update(SD1_ARC1_4)

    Sequence.update(SF1_STA1_1)
    Sequence.update(SF1_STA1_2)

    Sequence.update(ElSeparatorElement_entrance)

    ###################################### Get element from madx ######################################

    # element_list_from_madx, circumference_seqFile = generate_element_json(
    #     r"D:\AthenaLattice\Ion-Track-etched-Membrane\v9-3\itm.seq",
    # )
    # if (abs(circumference - circumference_seqFile)) > 1e-9:
    #     print(
    #         f"Error: Circumference from Beampara dict is = {circumference}, circumference from madx sequence file is {circumference_seqFile}. Check it!"
    #     )
    #     sys.exit(1)
    # for element in element_list_from_madx:
    #     # print(element)
    #     Sequence.update(element)

    ###################################### Space charge according to madx element ######################################

    # spacecharge_list = []
    # for ielem in element_list_from_madx:
    #     first_key = next(iter(ielem))
    #     name_elem = first_key
    #     s_elem = ielem[first_key]["S (m)"]
    #     l_elem = ielem[first_key]["L (m)"]
    #     drift_elem = ielem[first_key]["Drift length (m)"]
    #     command_elem = ielem[first_key]["Command"]

    #     dict_sc = {}

    #     # Aperture option:
    #     # Circle   : value = [radius] (in unit of m)
    #     # Rectangle: value = [half width, half height] (in unit of m)
    #     # Ellipse  : value = [horizontal semi axis, verticle semi axis] (in unit of m)
    #     # PIC mesh (refer to Rectangle): value = [] (empty, the aperture coincides with the PIC mesh)

    #     # Note1: if PIC mesh size is smaller than aperture, the PIC grid length will be increased to be able to cover the aperture
    #     # Note2: if the number of values in the value list is greater than the number required by the aperture, the redundant values will be ignored
    #     # Note3: for rectangle aperture, it is recommanded to align the PIC mesh with the aperture so as not to perform unnecessary calculations
    #     if command_elem == "SBendElement":
    #         dict_sc = {
    #             name_elem
    #             + "_sc": {
    #                 "S (m)": s_elem,
    #                 "Command": "SpaceCharge",
    #                 "Length (m)": l_elem + drift_elem,
    #                 "Number of PIC grid x": 256,
    #                 "Number of PIC grid y": 256,
    #                 "Grid x length": 1e-5,
    #                 "Grid y length": 3.5e-6,
    #                 "Aperture type": "Rectangle",
    #                 "Aperture value": [0.1, 0.1],
    #             }
    #         }
    #     elif command_elem == "QuadrupoleElement":
    #         dict_sc = {
    #             name_elem
    #             + "_sc": {
    #                 "S (m)": s_elem,
    #                 "Command": "SpaceCharge",
    #                 "Length (m)": l_elem + drift_elem,
    #                 "Number of PIC grid x": 512,
    #                 "Number of PIC grid y": 512,
    #                 "Grid x length": 2e-5,
    #                 "Grid y length": 5e-6,
    #                 "Aperture type": "Ellipse",
    #                 "Aperture value": [0.2, 0.1],
    #             }
    #         }
    #     else:
    #         dict_sc = {
    #             name_elem
    #             + "_sc": {
    #                 "S (m)": s_elem,
    #                 "Command": "SpaceCharge",
    #                 "Length (m)": l_elem + drift_elem,
    #                 "Number of PIC grid x": 256,
    #                 "Number of PIC grid y": 256,
    #                 "Grid x length": 1e-5,
    #                 "Grid y length": 3.5e-6,
    #                 "Aperture type": "PIC mesh",
    #                 "Aperture value": [],
    #             }
    #         }

    #     spacecharge_list.append(dict_sc)

    # # for dict_sc in spacecharge_list:
    # #     Sequence.update(dict_sc)
    # print(f"Number of space charge points: {len(spacecharge_list)}")

    ###################################### Oneturn map ######################################

    # lattice_oneturn_map = {
    #     "oneturn_map": {
    #         "S (m)": 0,
    #         "Command": "Twiss",
    #         "S previous (m)": 0,
    #         "Alpha x": 0,
    #         "Alpha y": 0,
    #         "Beta x (m)": 0.05,
    #         "Beta y (m)": 0.012,
    #         "Mu x": 0.3152,
    #         "Mu y": 0.3016,
    #         "Mu z": 0.0102,
    #         "Alpha x previous": 0,
    #         "Alpha y previous": 0,
    #         "Beta x previous (m)": 0.05,
    #         "Beta y previous (m)": 0.012,
    #         "Mu x previous": 0,
    #         "Mu y previous": 0,
    #         "Mu z previous": 0,
    #         # "Dx (m)": Dx[i],
    #         # "Dpx": Dpx[i],
    #         "longitudinal transfer": "matrix",
    #     },
    # }
    # Sequence.update(lattice_oneturn_map)

    ###################################### Distribution Monitor ######################################

    # Monitor to save bunch distribution
    Monitor_Dist_oneturn = {
        "DistMonitor_oneturn_0": {
            "S (m)": 0,
            "Command": "DistMonitor",
            "Save turns": [[1], [5000], [20000, 30000, 5000]],
        },
    }
    Sequence.update(Monitor_Dist_oneturn)

    Monitor_Dist_ES = {
        "DistMonitor_ES": {
            "S (m)": 26.84,
            "Command": "DistMonitor",
            "Save turns": [
                [100],
                [500],
                [1000, 5000, 1000],
            ],
        },
    }
    Sequence.update(Monitor_Dist_ES)

    ###################################### Statistic Monitor ######################################

    # Monitor to save bunch statistics
    Monitor_Stat_oneturn = {
        "StatMonitor_oneturn_0": {"S (m)": 0, "Command": "StatMonitor"},
    }
    Sequence.update(Monitor_Stat_oneturn)

    ###################################### RF Cavity ######################################

    # # RF cavity, length is 0, no drift
    # RF1 = {
    #     "RF_cavity1_"
    #     + str(circumference): {
    #         "S (m)": circumference,
    #         "Command": "RFElement",
    #         "RF Data files": ["D:path"],
    #     }
    # }
    # Sequence.update(RF1)

    ###################################### Sort and cut slice ######################################

    # Sort bunch at position s to realize bunch slicing
    for s_sort in np.linspace(0, circumference, 1, endpoint=False):
        sortPoint = {
            "SortBunch_"
            + str(s_sort): {
                "S (m)": s_sort,
                "Command": "SortBunch",
                "Sort purpose": "Space-charge",  # [Space-charge/Beam-beam]
            }
        }
        Sequence.update(sortPoint)

    ###################################### Particle Monitor ######################################

    # ParticleMonitor at position s to save specified particles
    PM_para = ParticleMonitorPara["Particle Monitor parameters"]
    s_PM = PM_para["Observer position S (m)"]
    for i in np.arange(len(s_PM)):
        partMoni = {
            "ParticleMonitor_"
            + str(s_PM[i]): {
                "S (m)": s_PM[i],
                "Command": "ParticleMonitor",
                "Observer Id": int(i),
            }
        }
        Sequence.update(partMoni)

    ###################################### Sort sequence by s ######################################

    Sequence = sort_sequence(Sequence)
    Sequencepara = {"Sequence": Sequence}

    ###################################### Write sequence to json file ######################################

    merged_dict = {
        **Beampara,
        **Spacecharge_sim_para,
        **BeambeamPara,
        **ParticleMonitorPara,
        **Sequencepara,
    }

    with open(config_path, "w") as jsonfile:
        json.dump(merged_dict, jsonfile, indent=4)

    ###################################### Finish ######################################

    print("Success")


if __name__ == "__main__":
    generate_simulation_config_beam0()
