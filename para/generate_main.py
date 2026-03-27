import numpy as np
import os
import json
import sys
import re

from toolkit import sort_sequence
from get_twiss_from_madx import get_twiss_from_madx_twissfile
from get_twiss_from_madx import get_twiss_interpolate_from_madx_twissfile
from get_element_from_madx import get_element_from_madx_twissfile
from get_element_from_madx import add_ramping_file
from generate_smooth_approx_twiss import generate_twiss_smooth_approximate


def generate_simulation_config_beam0(output_fileName="beam0.json"):
    """
    Generate a PASS simulation input file requires the following steps:
    1. set the global parameters: BeamPara (required), SpaceChargePara (optional), BeamBeamPara (optional)
    2. set the parameters of beam: Injection (required)
    3. set the ring sequence (at least include 1 of the following parts)
        (1) get full twiss from madx
        (2) get twiss from madx and interpolate it to wanted slices
        (3) get full element from madx
        (4) generate twiss by smooth approximation
        (5) generate one-turn map twiss
        Note: Element and Twiss can be mixed, but the transfer must comply the realistic physics
    4. add dynamic effect (optional) like space-charge, beam-beam, impedence and so on.
    5. add other elements (optional) like RF cavity, tune exciter, etc.
    6. add Monitors (optional) and Slice (optional) parameters .
    7. other personal processes (optional).
    """

    # ------------------------------------------------------------ Configure path (Do not change) ------------------------------------------------------------ #

    current_path, _ = os.path.split(__file__)
    parent_path = os.path.dirname(current_path)
    config_path = os.sep.join([parent_path, "para", output_fileName])
    print("The simulation configuration will be written to file: ", config_path)

    Sequence = {}

    # ------------------------------------------------------------ Step 1: Config global parameters ------------------------------------------------------------ #

    BeamPara = {
        "Name": "proton",  # [particle name]: arbitrary, just to let the user distinguish the beam
        "Number of protons per particle": 8,
        "Number of neutrons per particle": 10,  # if #neutrons is > 0, the mass of the particle is calculated based on nucleon mass
        "Number of charges per particle": 6,
        "Number of bunches per beam": 1,
        "Circumference (m)": 569.1,
        "GammaT": 7.635074035,
        "Number of turns": 200,  # number of simulate turns, start from 1 in program
        "Number of GPU devices": 1,
        "Device Id": [0],  # Nvidia GPU ID, if you have only 1 gpu, this value must be [0]
        "Output directory": "C:\\Users\\changmx\\Documents\\PassSimulation",
        "Is plot figure": False,
    }

    SpaceChargePara = {
        "Space-charge simulation parameters": {
            "Is enable space charge": False,
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

    # ------------------------------------------------------------ Step 2: Injection parameters ------------------------------------------------------------ #
    # Initialize all bunches' distribution of the beam
    Injection = {
        "Injection": {
            "S (m)": 0,
            "Command": "Injection",
            "bunch0": {
                "Kinetic energy per nucleon (eV/u)": 33.2e6,
                "Number of real particles per bunch": 3e11,
                "Number of macro particles per bunch": 1e6,
                "Mode": "1turn1time",  # [1turn1time/1turnxtime/xturnxtime]
                "Inject turns": [1],
                "Alpha x": -2.614303952,
                "Alpha y": 1.57442348,
                "Beta x (m)": 17.56341783,
                "Beta y (m)": 8.624482365,
                "Emittance x (m'rad)": 200e-6,
                "Emittance y (m'rad)": 100e-6,
                "Dx (m)": 0.0,
                "Dpx": 0.0,
                "Sigma z (m)": 0,
                "DeltaP/P": 0,
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
                "Is load distribution": False,
                "Name of loaded file":
                "1455_56_beam0_proton_bunch0_1000000_hor_kv_longi_uniform_Dx_0.000000_injection.csv",  # file must be put in "Output directory/distribution/fixed/"
                "Is save initial distribution": True,
                "Insert particle coordinate": [[0, 0, 0, 0, 0, 0]],
            },
        }
    }
    Sequence.update(Injection)

    # ------------------------------------------------------------ Step 3: Interpolate twiss from madx ------------------------------------------------------------ #

    # twiss_dict_interp, circumference_interp = get_twiss_interpolate_from_madx_twissfile(
    #     twiss_file_path=r"C:\Users\changmx\Documents\PASS\para\bring.tfs",
    #     error_file_path=r"C:\Users\changmx\Documents\PASS\para\error_sextupoleerror.tfs",
    #     num_interp_slice=100,
    #     logi_transfer_method="off",
    #     muz=0.001,
    #     DQx=2.0,
    #     DQy=3.0,
    #     is_field_error=True,
    #     insert_element_name_pattern=[],
    #     interp_kind="cubic",
    # )
    # BeamPara["Circumference (m)"] = circumference_interp

    # Sequence.update(twiss_dict_interp)

    # ------------------------------------------------------------ Step 3: Get twiss from madx ------------------------------------------------------------ #

    # twiss_dict_from_madx, circumference_twissFile = get_twiss_from_madx_twissfile(
    #     twiss_file_path=r"C:\Users\changmx\Documents\PASS\para\bring.tfs",
    #     error_file_path=r"C:\Users\changmx\Documents\PASS\para\error_sextupoleerror.tfs",
    #     logi_transfer_method="off",
    #     muz=0.001,
    #     DQx=0,
    #     DQy=0,
    #     is_field_error=False,
    #     insert_element_name_pattern=["BRMG41Q2222"],
    # )
    # BeamPara["Circumference (m)"] = circumference_twissFile

    # Sequence.update(twiss_dict_from_madx)

    # ------------------------------------------------------------ Step 3: Get element from madx ------------------------------------------------------------ #

    element_dict_from_madx, circumference_from_madx = get_element_from_madx_twissfile(
        twiss_file_path=r"C:\Users\changmx\Documents\PASS\para\bring.tfs",
        error_file_path=r"C:\Users\changmx\Documents\PASS\para\error_sextupoleerror.tfs",
        is_merge_drift=True,
        is_field_error=False,
    )
    BeamPara["Circumference (m)"] = circumference_from_madx

    Sequence.update(element_dict_from_madx)

    # add_ramping_file(
    #     Sequence,
    #     elem_name_re_pattern=["BRMG41SH"],
    #     file_key="K2L ramping file",
    #     file_path=r"C:\Users\changmx\Documents\PASS\para\k2l_ramping.csv",
    # )

    # ------------------------------------------------------------ Step 3: Get twiss from smooth approximation ------------------------------------------------------------ #

    # twiss_dict_smooth_approx, circumference_smooth_approx = generate_twiss_smooth_approximate(
    #     circum=BeamPara["Circumference (m)"],
    #     Qx=9.47,
    #     Qy=9.43,
    #     numPoints=100,
    #     logi_transfer="off",
    #     muz=0.0123,
    # )

    # Sequence.update(twiss_dict_smooth_approx)

    # ------------------------------------------------------------ Step 3: Oneturn map ------------------------------------------------------------ #

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

    # ------------------------------------------------------------ Step 4: Space charge ------------------------------------------------------------ #

    # Aperture option:
    #   Circle   : value = [radius] (in unit of m)
    #   Rectangle: value = [half width, half height] (in unit of m)
    #   Ellipse  : value = [horizontal semi axis, verticle semi axis] (in unit of m)
    #   PIC mesh (refer to Rectangle): value = [] (empty, the aperture coincides with the PIC mesh)

    #   Note1: if PIC mesh size is smaller than aperture, the PIC grid length will be increased to be able to cover the aperture
    #   Note2: if the number of values in the value list is greater than the number required by the aperture, the redundant values will be ignored
    #   Note3: for rectangle aperture, it is recommanded to align the PIC mesh with the aperture so as not to perform unnecessary calculations

    # sc_dict = {}
    # sc_point_pattern = ["twiss"]

    # combined_pattern = re.compile('|'.join(f'({pattern.lower()})' for pattern in sc_point_pattern))
    # sort_sequence(Sequence)

    # sc_count = 0

    # for key, value in Sequence.items():
    #     # print(key)
    #     is_match = combined_pattern.search(key.lower())

    #     if is_match:
    #         sc_count += 1
    #         sc_dict[f"SC[{sc_count}]_{key}"] = {
    #             "S (m)": value["S (m)"],
    #             "Command": "SpaceCharge",
    #             "L (m)": 0,
    #             "Aperture type": "Rectangle",  # [Circle/Rectangle/Ellipse]
    #             "Aperture value": [
    #                 0.2,
    #                 0.2,
    #             ],  # [Circle: radius/Rectangle:half width, half height/Ellipse:a,b]
    #             "Number of PIC grid x": 200,
    #             "Number of PIC grid y": 200,
    #             "Grid x length": 0.002,
    #             "Grid y length": 0.002,
    #         }
    #         # Note: If you want, you can configure different settings such as Ngrid, GridL,
    #         #       and aperture values for different elements or different beam envelopes.
    #         #       You only need to modify the if condition.

    # sc_keys = list(sc_dict.keys())
    # for i in range(len(sc_keys)):
    #     if i == 0:
    #         sc_dict[sc_keys[0]]["L (m)"] = sc_dict[sc_keys[0]]["S (m)"] + (BeamPara["Circumference (m)"] - sc_dict[sc_keys[-1]]["S (m)"])
    #     else:
    #         sc_dict[sc_keys[i]]["L (m)"] = sc_dict[sc_keys[i]]["S (m)"] - sc_dict[sc_keys[i - 1]]["S (m)"]

    # sc_length_counts = 0
    # sc_delete_keys = []
    # for key, value in sc_dict.items():
    #     if value["L (m)"] > 1e-10:
    #         sc_length_counts += value["L (m)"]
    #     else:
    #         sc_delete_keys.append(key)
    # if len(sc_delete_keys) > 0:
    #     for sc_del_key in sc_delete_keys:
    #         sc_dict.pop(sc_del_key)
    #     print(f"[Space Charge] The length of these sc point is 0, we have delete them: {sc_delete_keys}")

    # if abs(sc_length_counts - BeamPara["Circumference (m)"]) < 1e-6:
    #     print(
    #         f"[Space Charge] Pass the circumference test: theory = {BeamPara['Circumference (m)']} m, current = {sc_length_counts} m, diff = {sc_length_counts - BeamPara['Circumference (m)']:.15e} m"
    #     )
    # else:
    #     print(
    #         f"[Space Charge] Failed the circumference test: theory = {BeamPara['Circumference (m)']} m, current = {sc_length_counts} m, diff = {sc_length_counts - BeamPara['Circumference (m)']:.15e} m"
    #     )
    #     sys.exit(1)

    # print(f"[Space Charge] success: {len(sc_dict)} space charge points has been generated.")

    # Sequence.update(sc_dict)

    # ------------------------------------------------------------ Step 5: Input single element ------------------------------------------------------------ #

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

    # ------------------------------------------------------------ Step 5: RF Cavity ------------------------------------------------------------ #

    # RF1 = {
    #     "RF_cavity1_"
    #     + str(0): {
    #         "S (m)": 0,
    #         "Command": "RFElement",
    #         "DeltaP/P aperture": [-0.005, 0.005],
    #         "RF Data files": [r"D:\PASS\para\rf_data.csv"],
    #     }
    # }
    # Sequence.update(RF1)

    # ------------------------------------------------------------ Step 5: Tune Exciter ------------------------------------------------------------ #

    # TuneExciter1 = {
    #     "TuneExciter1": {
    #         "S (m)": 0,
    #         "Command": "TuneExciterElement",
    #         "Kick angle (rad)": 100e-6,
    #         "Frequency center (Hz)": 64394.87000039443,
    #         "Scan period (s)": 5e-3,
    #         "Scan frequency range (Hz)": 100,
    #         "Kick direction": "x",
    #         "Turns": [1000, 4000],
    #     }
    # }
    # Sequence.update(TuneExciter1)

    # ------------------------------------------------------------ Step 6: Distribution Monitor ------------------------------------------------------------ #
    # Monitor to save bunch distribution at specific position

    # Monitor_Dist1 = {
    #     "DistMonitor_start": {
    #         "S (m)": 0,
    #         "Command": "DistMonitor",
    #         "Save turns": [
    #             [1],
    #             [2],
    #             [1000, 5000, 1000],
    #             [6000],
    #             [8000],
    #             [20000, 30000, 5000],
    #         ],
    #     },
    # }
    # Sequence.update(Monitor_Dist1)

    # ------------------------------------------------------------ Step 6: Statistic Monitor ------------------------------------------------------------ #
    # Monitor to save bunch statistics

    Monitor_Stat1 = {
        "StatMonitor_start": {
            "S (m)": 0,
            "Command": "StatMonitor"
        },
    }
    Sequence.update(Monitor_Stat1)

    Monitor_Stat2 = {
        "StatMonitor_2": {
            "S (m)": 83.57124735,
            "Command": "StatMonitor"
        },
    }
    Sequence.update(Monitor_Stat2)

    # ------------------------------------------------------------ Step 6: Phase Monitor ------------------------------------------------------------ #
    # Monitor to save phase advance

    # PhaseMonitor = {
    #     "PhaseMonitor_0": {
    #         "S (m)": 0,
    #         "Command": "PhaseMonitor",
    #         "Is enable phase monitor": True,
    #         "Beta x (m)": Injection["Injection"]["bunch0"]["Beta x (m)"],
    #         "Beta y (m)": Injection["Injection"]["bunch0"]["Beta y (m)"],
    #         "Alpha x": Injection["Injection"]["bunch0"]["Alpha x"],
    #         "Alpha y": Injection["Injection"]["bunch0"]["Alpha y"],
    #         "Save turns": [[1, 100], [1000, 10000, 1000, 50]],
    #     }
    # }
    # Sequence.update(PhaseMonitor)

    # ------------------------------------------------------------ Step 6:  Particle Monitor ------------------------------------------------------------ #

    # ParticleMonitor at position s to save specified particles

    # ParticleMonitorPara = {
    #     "Particle Monitor parameters": {
    #         "Is enable particle monitor": False,
    #         "Number of particles to save": 3,
    #         "Save turn range": [1, 1000, 1],
    #         "Observer position S (m)": [0],
    #     }
    # }
    # ParticleMonitor_dict = {}
    # ParticleMonitor_obs = ParticleMonitorPara["Particle Monitor parameters"]["Observer position S (m)"]
    # num_obs = len(ParticleMonitor_obs)
    # for i in np.arange(num_obs):
    #     ParticleMonitor_dict[f"ParticleMonitor[{i+1}]"] = {
    #         "S (m)": ParticleMonitor_obs[i],
    #         "Command": "ParticleMonitor",
    #         "Observer Id": int(i),
    #     }
    # Sequence.update(ParticleMonitor_dict)

    # ------------------------------------------------------------ Step 6: Sort and cut slice ------------------------------------------------------------ #
    # Sort bunch at position s to realize bunch slicing

    # sortPoint_dict = {}
    # num_sort = 1
    # s_sort = np.linspace(0, BeamPara["Circumference (m)"], num_sort, endpoint=False)
    # for i in range(len(s_sort)):
    #     sortPoint_dict[f"SortBunch[{i+1}]"] = {
    #         "S (m)": s_sort[i],
    #         "Command": "SortBunch",
    #         "Sort purpose": "Space-charge",  # [Space-charge/Beam-beam]
    #     }
    # Sequence.update(sortPoint_dict)

    # ------------------------------------------------------------ Step 7: User's customized section ------------------------------------------------------------ #

    # Do whateve you want.
    # test_element = {
    #     "test_quad": {
    #         "S (m)": 1,
    #         "Command": "QuadrupoleElement",
    #         "L (m)": 0,
    #         "K1L (m^-1)": 0.2,
    #         "K1SL (m^-1)": 0.0,
    #         "Is field error": False,
    #         "Field error KNL": [],
    #         "Field error KSL": [],
    #         "Is ramping": True,
    #         "K1L ramping file": r"C:\Users\changmx\Documents\PASS\para\k1l.csv",
    #         "K1SL ramping file": "",
    #     }
    # }
    # Sequence.update(test_element)

    # ------------------------------------------------------------ Finally (Do not change): Sort sequence by s ------------------------------------------------------------ #

    Sequence = sort_sequence(Sequence)
    SequencePara = {"Sequence": Sequence}

    # ------------------------------------------------------------ Finally (Do not change): Write sequence to json file ------------------------------------------------------------ #

    try:
        merged_dict = {
            **BeamPara,
            **SpaceChargePara,
            **BeamBeamPara,
            **ParticleMonitorPara,
            **SequencePara,
        }
    except:
        merged_dict = {
            **BeamPara,
            **SpaceChargePara,
            **BeamBeamPara,
            **SequencePara,
        }

    with open(config_path, "w") as jsonfile:
        json.dump(merged_dict, jsonfile, indent=4)

    # ------------------------------------------------------------ Finally (Do not change): Finished ------------------------------------------------------------ #

    print("Success")


if __name__ == "__main__":
    generate_simulation_config_beam0()
