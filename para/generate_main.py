import numpy as np
import os
import json
import sys
from collections import OrderedDict

from get_twiss_from_madx import gen_twiss_from_madx

from get_element_from_madx import get_element_from_madx
from generate_smooth_approx_twiss import generate_twiss_smooth_approximate


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
        "SortBunch": 100,  # 次级优先级
        "Twiss": 200,
        "MarkerElement": 300,
        "SBendElement": 300,
        "RBendElement": 300,
        "QuadrupoleElement": 300,
        "SextupoleNormElement": 300,
        "SextupoleSkewElement": 300,
        "OctupoleElement": 300,
        "HKickerElement": 300,
        "VKickerElement": 300,
        "RFElement": 300,
        "ElSeparatorElement": 300,
        "SpaceCharge": 400,
        "WakeField": 500,
        "BeamBeam": 600,
        "ElectronCloud": 700,
        "LumiMonitor": 800,
        "PhaseMonitor": 800,
        "DistMonitor": 800,
        "StatMonitor": 800,
        "ParticleMonitor": 800,
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

    # ------------------------------------------------------------ Config global parameters ------------------------------------------------------------ #

    BeamPara = {
        "Name": "proton",  # [particle name]: arbitrary, just to let the user distinguish the beam
        "Number of protons per particle": 1,
        "Number of neutrons per particle": 0,  # if #neutrons is > 0, the mass of the particle is calculated based on nucleon mass
        "Number of charges per particle": 1,
        "Number of bunches per beam": 1,
        "Circumference (m)": 569.1,
        "GammaT": 7.635074035,
        "Number of turns": 48500,
        "Number of GPU devices": 1,
        "Device Id": [0],
        "Output directory": "D:\\PassSimulation",
        "Is plot figure": False,
    }

    circumference = BeamPara["Circumference (m)"]

    SpaceChargePara = {
        "Space-charge simulation parameters": {
            "Is enable space charge": True,
            "Number of slices": 10,
            "Slice model": "Equal particle",  # [Equal particle/Equal length]
            "Field solver": "PIC_FD_CUDSS",  # [PIC_FD_CUDSS/PIC_Conv/PIC_FD_AMGX/PIC_FD_FFT/Eq_Quasi_Static/Eq_Frozen]
        }
    }

    BeamBeamPara = {
        "Beam-beam simulation parameters": {
            "Is enable beam-beam": False,
            "Number of slices": 10,
            "Slice model": "Equal particle",  # [Equal particle/Equal length]
            "Field solver": "PIC_Conv",  # [PIC_Conv/PIC_FD_AMGX/PIC_FD_FFT/Eq_Quasi_Static/Eq_Frozen]
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
            "Number of particles to save": 3,
            "Save turn range": [1, 1000, 1],
            # "Observer position S (m)": [0, 26.84],
            "Observer position S (m)": [0],
        }
    }

    # -------------------------------------------------------------- Start create sequence ------------------------------------------------------------- #

    Sequence = {}

    # -------------------------------------------------------------- Injection parameters -------------------------------------------------------------- #
    # Initialize all bunches' distribution of the beam
    Injection = {
        "Injection": {
            "S (m)": 0,
            "Command": "Injection",
            "bunch0": {
                "Kinetic energy per nucleon (eV/u)": 48e6,
                "Number of real particles per bunch": 3e11,
                "Number of macro particles per bunch": 1e4,
                "Mode": "1turn1time",  # [1turn1time/1turnxtime/xturnxtime]
                "Inject turns": [1],
                "Alpha x": -2.6143039521482168,
                "Alpha y": 1.5744234799406003,
                "Beta x (m)": 17.563417831999914,
                "Beta y (m)": 8.624482364741164,
                "Emittance x (m'rad)": 12.5e-6,
                "Emittance y (m'rad)": 6.25e-6,
                "Dx (m)": 0.0,
                "Dpx": 0.0,
                "Sigma z (m)": 569.0984841047994,
                "DeltaP/P": 6.66667e-4,
                "Transverse dist": "kv",  # [kv/gaussian/uniform]
                "Longitudinal dist": "uniform",  # [gaussian/uniform]
                "Offset x": {
                    "Is offset": False,
                    "Offset (m)": 5e-3,
                },
                "Offset y": {
                    "Is offset": False,
                    "Offset (m)": 0,
                },
                "Is load distribution": True,
                "Name of loaded file": "wangl_logi.csv",  # file must be put in "Output directory/distribution/fixed/"
                "Is save initial distribution": True,
                "Insert particle coordinate": [
                    [0, 0, 0, 0, 50.2023089662953, 0.000400381857834703]
                ],
            },
        }
    }
    Sequence.update(Injection)

    # -------------------------------------------------------------- Get twiss  from madx -------------------------------------------------------------- #

    # twiss_list_from_madx, circumference_twissFile = gen_twiss_from_madx(
    #     seq_file=r"D:\PASS\para\BRingOptics.seq",
    #     # error_file=r"D:\PASS\para\error.madx",
    #     seq_name="ring",
    #     logi_transfer_method="drift",
    #     muz=0,
    #     DQx=-11.62259943,
    #     DQy=-11.35280561,
    #     centre=True,
    #     is_add_sextupole=True,
    # )
    # if (abs(circumference - circumference_twissFile)) > 1e-10:
    #     print(
    #         f"Warning: Circumference from BeamPara dict is = {circumference}, circumference from madx twiss file is {circumference_twissFile}."
    #     )
    #     BeamPara["Circumference (m)"] = circumference_twissFile
    #     circumference = BeamPara["Circumference (m)"]
    #     print(f"Circumference from BeamPara dict has been changed to {circumference}")

    # for twiss in twiss_list_from_madx:
    #     Sequence.update(twiss)

    # ------------------------------------------------------- Space charge according to madx twiss ----------------------------------------------------- #

    # spacecharge_list = []
    # for ielem in twiss_list_from_madx:
    #     first_key = next(iter(ielem))
    #     name_sc = first_key
    #     command_elem = ielem[first_key]["Command"]

    #     if command_elem == "Twiss":
    #         s_sc = ielem[first_key]["S (m)"]
    #         s_pre_sc = ielem[first_key]["S previous (m)"]

    #         dict_sc = {
    #             name_sc
    #             + "_sc": {
    #                 "S (m)": s_sc,
    #                 "Command": "SpaceCharge",
    #                 "Length (m)": s_sc - s_pre_sc,
    #                 "Aperture type": "Rectangle",  # [Circle/Rectangle/Ellipse]
    #                 "Aperture value": [
    #                     0.2,
    #                     0.2,
    #                 ],  # [Circle: radius/Rectangle:half width, half height/Ellipse:a,b]
    #                 "Number of PIC grid x": 200,
    #                 "Number of PIC grid y": 200,
    #                 "Grid x length": 0.002,
    #                 "Grid y length": 0.002,
    #             }
    #         }
    #         spacecharge_list.append(dict_sc)

    # for dict_sc in spacecharge_list:
    #     Sequence.update(dict_sc)
    # print(f"Number of space charge points: {len(spacecharge_list)}")

    # -------------------------------------------------------------- Get element from madx ------------------------------------------------------------- #

    # element_list_from_madx, circumference_seqFile = get_element_from_madx(
    #     seq_file=r"D:\PASS\para\BRingOptics.seq",
    #     # error_file=r"D:\PASS\para\error.madx",
    #     seq_name="ring",
    # )
    # if (abs(circumference - circumference_seqFile)) > 1e-10:
    #     print(
    #         f"Warning: Circumference from BeamPara dict is = {circumference}, circumference from madx twiss file is {circumference_seqFile}."
    #     )
    #     BeamPara["Circumference (m)"] = circumference_seqFile
    #     circumference = BeamPara["Circumference (m)"]
    #     print(f"Circumference from BeamPara dict has been changed to {circumference}")
    # for element in element_list_from_madx:
    #     # print(element)
    #     Sequence.update(element)

    # ----------------------------------------------------- Space charge according to madx element ----------------------------------------------------- #

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

    # ------------------------------------------------------- Get twiss from smooth approximation ------------------------------------------------------ #

    # twiss_list_smooth_approx, circumference_smooth_approx = (
    #     generate_twiss_smooth_approximate(
    #         circum=569.1,
    #         mux=9.47,
    #         muy=9.43,
    #         numPoints=100,
    #         logi_transfer="off",
    #         muz=0.0123,
    #     )
    # )
    # if (abs(circumference - circumference_smooth_approx)) > 1e-9:
    #     print(
    #         f"Error: Circumference from BeamPara dict is = {circumference}, circumference from smooth approximate file is {circumference_smooth_approx}. Check it!"
    #     )
    #     sys.exit(1)
    # for twiss in twiss_list_smooth_approx:
    #     Sequence.update(twiss)

    # ----------------------------------------------- Space charge according to smooth approximate twiss ----------------------------------------------- #

    # spacecharge_list = []
    # for ielem in twiss_list_smooth_approx:
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
    #             "Aperture type": "Rectangle",  # [Circle/Rectangle/Ellipse]
    #             "Aperture value": [
    #                 0.2,
    #                 0.2
    #             ],  # [Circle: radius/Rectangle:half width, half height/Ellipse:a,b]
    #             "Number of PIC grid x": 200,
    #             "Number of PIC grid y": 200,
    #             "Grid x length": 0.002,
    #             "Grid y length": 0.002,
    #         }
    #     }
    #     spacecharge_list.append(dict_sc)

    # for dict_sc in spacecharge_list:
    #     Sequence.update(dict_sc)
    # print(f"Number of space charge points: {len(spacecharge_list)}")

    # ----------------------------------------------------------- Input single Space charge ------------------------------------------------------------ #

    # sh_list = []
    # sv_list = []
    # sh_list.append(elem_dict[name] = {
    #                 "S (m)": s,
    #                 "Command": class_map["sextupole norm"],
    #                 "L (m)": l,
    #                 "Drift length (m)": 0,
    #                 "k2 (m^-3)": seq[i].k2,
    #                 "isFieldError": False,
    #                 "Error order": 0,
    #                 "KNL": [],
    #                 "KSL": [],
    #                 "Is thin lens": False,
    #             })
    # SC1 = {
    #     "SpaceCharge_0.1": {
    #         "S (m)": 0.1,
    #         "Command": "SpaceCharge",
    #         "Length (m)": 0.5,
    #         "Aperture type": "Rectangle",  # [Circle/Rectangle/Ellipse]
    #         "Aperture value": [
    #             0.1,
    #             0.1,
    #         ],  # [Circle: radius/Rectangel:half width, half height/Ellipse:a,b]
    #         "Number of PIC grid x": 100,
    #         "Number of PIC grid y": 100,
    #         "Grid x length": 0.002,
    #         "Grid y length": 0.002,
    #     }
    # }

    # Sequence.update(SC1)

    # -------------------------------------------------------------- Input single element -------------------------------------------------------------- #

    # SF1_ARC1_1 = {
    #     "sf1_arc1_1": {
    #         "S (m)": 8.37,
    #         "Command": "SextupoleNormElement",
    #         "L (m)": 0.2,
    #         "Drift length (m)": 0,
    #         "k2 (m^-3)": 2.768917952,
    #         "isFieldError": False,
    #         "isIgnoreLength": True,
    #     },
    # }

    # ElSeparatorElement_entrance = {
    #     "ES_26.84": {
    #         "Command": "ElSeparatorElement",
    #         "S (m)": 26.84,
    #         "ES Horizontal position (m)": 0.04,
    #     }
    # }

    # Sequence.update(SF1_ARC1_1)

    # Sequence.update(ElSeparatorElement_entrance)

    # ------------------------------------------------------------------ Oneturn map ------------------------------------------------------------------- #

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
    lattice_oneturn_map = {
        "ring$end:1_569.0984841047994": {
            "S (m)": 569.0984841047994,
            "Command": "Twiss",
            "S previous (m)": 0,
            "Alpha x": -2.6143039521481892,
            "Alpha y": 1.5744234799406198,
            "Beta x (m)": 17.563417831999786,
            "Beta y (m)": 8.624482364741272,
            "Mu x": 0,
            "Mu y": 0,
            "Mu z": 0.0,
            "Dx (m)": -7.906159725352051e-10,
            "Dpx": 2.433671012036641e-10,
            "Alpha x previous": -2.6143039521481892,
            "Alpha y previous": 1.5744234799406198,
            "Beta x previous (m)": 17.563417831999786,
            "Beta y previous (m)": 8.624482364741272,
            "Mu x previous": 0.47,
            "Mu y previous": 0.43,
            "Mu z previous": 0.0,
            "Dx (m) previous": 0,
            "Dpx previous": 0,
            "DQx": 0,
            "DQy": 0,
            "Longitudinal transfer": "drift",
        }
    }
    Sequence.update(lattice_oneturn_map)

    # -------------------------------------------------------------- Distribution Monitor -------------------------------------------------------------- #

    # Monitor to save bunch distribution
    Monitor_Dist_oneturn = {
        "DistMonitor_oneturn_0": {
            "S (m)": 0,
            "Command": "DistMonitor",
            "Save turns": [[1], [2], [5000], [6000], [8000], [20000, 30000, 5000]],
        },
    }
    Sequence.update(Monitor_Dist_oneturn)

    # DistMonitor2 = {
    #     "DistMonitor_end": {
    #         "S (m)": 569.0984841048011,
    #         "Command": "DistMonitor",
    #         "Save turns": [[1], [2]],
    #     },
    # }
    # Sequence.update(DistMonitor2)

    # Monitor_Dist_ES = {
    #     "DistMonitor_ES": {
    #         "S (m)": 26.84,
    #         "Command": "DistMonitor",
    #         "Save turns": [
    #             [100],
    #             [500],
    #             [1000, 5000, 1000],
    #         ],
    #     },
    # }
    # Sequence.update(Monitor_Dist_ES)

    # --------------------------------------------------------------- Statistic Monitor ---------------------------------------------------------------- #

    # Monitor to save bunch statistics
    Monitor_Stat_oneturn = {
        "StatMonitor_oneturn_0": {"S (m)": 0, "Command": "StatMonitor"},
    }
    Sequence.update(Monitor_Stat_oneturn)
    # Monitor_2 = {
    #     "Monitor2": {"S (m)": 24.580000000000013, "Command": "StatMonitor"},
    # }
    # Sequence.update(Monitor_2)

    # ----------------------------------------------------------------- Phase Monitor ------------------------------------------------------------------ #

    # Monitor to save phase advance
    PhaseMonitor = {
        "PhaseMonitor_0": {
            "S (m)": 0,
            "Command": "PhaseMonitor",
            "Is enable phase monitor": True,
            "Beta x (m)": Injection["Injection"]["bunch0"]["Beta x (m)"],
            "Beta y (m)": Injection["Injection"]["bunch0"]["Beta y (m)"],
            "Alpha x": Injection["Injection"]["bunch0"]["Alpha x"],
            "Alpha y": Injection["Injection"]["bunch0"]["Alpha y"],
            "Save turns": [[1, 100], [1000, 10000, 1000, 50]],
        }
    }
    Sequence.update(PhaseMonitor)

    # ------------------------------------------------------------------- RF Cavity -------------------------------------------------------------------- #

    # RF cavity, length is 0, no drift
    RF1 = {
        "RF_cavity1_"
        + str(0): {
            "S (m)": 0,
            "Command": "RFElement",
            "DeltaP/P aperture": [-0.005, 0.005],
            "RF Data files": [r"D:\PASS\para\rf_data.csv"],
        }
    }
    Sequence.update(RF1)

    # --------------------------------------------------------------- Sort and cut slice --------------------------------------------------------------- #

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

    # --------------------------------------------------------------- Particle Monitor ----------------------------------------------------------------- #

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

    # --------------------------------------------------------------- Sort sequence by s --------------------------------------------------------------- #

    Sequence = sort_sequence(Sequence)
    SequencePara = {"Sequence": Sequence}

    # ---------------------------------------------------------- Write sequence to json file ----------------------------------------------------------- #

    merged_dict = {
        **BeamPara,
        **SpaceChargePara,
        **BeamBeamPara,
        **ParticleMonitorPara,
        **SequencePara,
    }

    with open(config_path, "w") as jsonfile:
        json.dump(merged_dict, jsonfile, indent=4)

    # ------------------------------------------------------------------- Finished --------------------------------------------------------------------- #

    print("Success")


if __name__ == "__main__":
    generate_simulation_config_beam0()
