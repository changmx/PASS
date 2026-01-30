import numpy as np
import sys
import os
import re
from collections import Counter
from scipy import interpolate
import matplotlib.pyplot as plt

import tfs  # tfs_pandas, https://pylhc.github.io/tfs/
from get_error_from_madx import get_field_error_from_madx_errorfile
from toolkit import class_map


def get_twiss_from_madx_twissfile(
    twiss_file_path,
    error_file_path="",
    logi_transfer_method="off",
    muz=0.0,
    DQx="from_file",
    DQy="from_file",
    is_field_error=False,
    insert_element_name_pattern=[],
):
    """
    twiss_file_path: str, path of madx generated twiss file
    error_file_path: str, path of madx generated error file
    logi_transfer_method: str, "off"/"drift"/"matrix"
    muz: float, longitudinal tune
    DQx: str or float, if DQx is str("from_file"), DQx is equal to the value in twiss file;
                       if DQx is float, DQx is equal to the setting value.
    DQy: str or float, if DQy is str("from_file"), DQy is equal to the value in twiss file;
                       if DQy is float, DQy is equal to the setting value.is_field_error: 说明
    is_field_error: bool, whether add field error from error file to elem_dict as multipole style
    insert_element_name_pattern: list of str, save the regular expression used for element specific_name matching.
                                 if matching successfully, the element will be inserted as thin-lens to elem_dict.
                                 only support thin-lens quad. setx. oct. multi.
    """

    # ------------------------- Read madx twiss file ------------------------- #

    twiss_table = tfs.read(twiss_file_path)  # TFSDataFrame, which is DataFrame + headers

    headers = twiss_table.headers  # get header information such as particle, energy, etc.
    column_names = twiss_table.columns  # get column names such as NAME, S, BETX, etc.
    shape = twiss_table.shape  # get data shape (rows, columns)

    num_elem = shape[0]
    print(
        f"[Get \033[31mTwiss\033[0m From Madx] There are '{num_elem}' elements in twiss file, the first and last element names are '{twiss_table.iloc[0]['NAME']}' and '{twiss_table.iloc[-1]['NAME']}' respectively."
    )
    circumference = headers["LENGTH"]
    Qx = headers["Q1"]
    Qy = headers["Q2"]
    DQx_file = headers["DQ1"]
    DQy_file = headers["DQ2"]

    if DQx == "from_file":
        DQx = DQx_file
    if DQy == "from_file":
        DQy = DQy_file

    if abs(DQx - DQx_file) > 1e-10:
        print(f"[Get \033[31mTwiss\033[0m From Madx] Warning: DQx in twiss file is {DQx_file}, but the current setting is {DQx}")
    if abs(DQy - DQy_file) > 1e-10:
        print(f"[Get \033[31mTwiss\033[0m From Madx] Warning: DQy in twiss file is {DQy_file}, but the current setting is {DQy}")

    print(f"[Get \033[31mTwiss\033[0m From Madx] Circumference = {circumference}, Qx = {Qx}, Qy ={Qy}, DQx = {DQx}, DQy = {DQy}")

    # ------------------ Generate PASS required twiss data ---------------- #

    elem_dict = {}
    elem_name_list = []
    re_match_record = []

    betx = twiss_table["BETX"]
    bety = twiss_table["BETY"]
    alfx = twiss_table["ALFX"]
    alfy = twiss_table["ALFY"]
    Dx = twiss_table["DX"]
    Dpx = twiss_table["DPX"]
    mux = twiss_table["MUX"]
    muy = twiss_table["MUY"]

    s = twiss_table["S"]
    l = twiss_table["L"]

    for i in np.arange(num_elem):
        elem_name = twiss_table.iloc[i]["NAME"]
        elem_type = twiss_table.iloc[i]["KEYWORD"]

        elem_name_list.append(elem_name)
        elem_count_result = Counter(elem_name_list)
        elem_appear_times = elem_count_result[elem_name]
        specific_name = f"{elem_name}[{elem_appear_times}]"

        if i == 0:
            elem_dict[specific_name] = {
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
            }
        else:
            elem_dict[specific_name] = {
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
                "DQx": DQx * (mux[i] - mux[i - 1]) / Qx,
                "DQy": DQy * (muy[i] - muy[i - 1]) / Qy,
                # "DQx": DQx * (s[i] - s[i - 1]) / circumference,
                # "DQy": DQy * (s[i] - s[i - 1]) / circumference,
                "Longitudinal transfer": logi_transfer_method,
            }

    print(f"[Get \033[31mTwiss\033[0m From Madx] Success: {len(elem_dict)} twiss elements have been read from twiss file")

    # ------------------ Insert element ------------------ #

    if len(insert_element_name_pattern) > 0:
        combined_pattern = re.compile('|'.join(f'({pattern})' for pattern in insert_element_name_pattern))

        for i in np.arange(num_elem):
            elem_name = twiss_table.iloc[i]["NAME"]
            elem_type = twiss_table.iloc[i]["KEYWORD"]

            elem_name_list.append(elem_name)
            elem_count_result = Counter(elem_name_list)
            elem_appear_times = elem_count_result[elem_name]
            specific_name = f"{elem_name}[{elem_appear_times}]"

            is_match = combined_pattern.search(specific_name)

            if is_match:
                specific_name_insert = f"{specific_name}_insert"
                re_match_record.append(specific_name_insert)

                if elem_type.lower() == "quadrupole":
                    elem_dict[specific_name_insert] = {
                        "S (m)": twiss_table.iloc[i]["S"],
                        "Command": class_map["quadrupole"],
                        "L (m)": 0,
                        "K1L (m^-1)": twiss_table.iloc[i]["K1L"],
                        "K1SL (m^-1)": twiss_table.iloc[i]["K1SL"],
                        "Is field error": False,
                        "Field error KNL": [],
                        "Field error KSL": [],
                        "Is ramping": False,
                        "K1L ramping file": "",
                        "K1SL ramping file": "",
                    }
                elif elem_type.lower() == "sextupole":
                    elem_dict[specific_name_insert] = {
                        "S (m)": twiss_table.iloc[i]["S"],
                        "Command": class_map["sextupole"],
                        "L (m)": 0,
                        "K2L (m^-2)": twiss_table.iloc[i]["K2L"],
                        "K2SL (m^-2)": twiss_table.iloc[i]["K2SL"],
                        "Is field error": False,
                        "Field error KNL": [],
                        "Field error KSL": [],
                        "Is ramping": False,
                        "K2L ramping file": "",
                        "K2SL ramping file": "",
                    }
                elif elem_type.lower() == "octupole":
                    elem_dict[specific_name_insert] = {
                        "S (m)": twiss_table.iloc[i]["S"],
                        "Command": class_map["Octupole"],
                        "L (m)": 0,
                        "K3L (m^-3)": twiss_table.iloc[i]["K3L"],
                        "K3SL (m^-3)": twiss_table.iloc[i]["K3SL"],
                        "Is field error": False,
                        "Field error KNL": [],
                        "Field error KSL": [],
                        "Is ramping": False,
                        "K3L ramping file": "",
                        "K3SL ramping file": "",
                    }
                elif elem_type.lower() == "multipole":
                    elem_dict[specific_name_insert] = {
                        "S (m)": twiss_table.iloc[i]["S"],
                        "Command": class_map["multipole"],
                        "L (m)": 0,
                        "KiL": [],
                        "KiSL": [],
                        "Is ramping": False,
                        "KL ramping file": "",
                    }
                elif elem_type.lower() == "hkicker" or elem_type.lower() == "vkicker" or elem_type.lower() == "kicker" or elem_type.lower(
                ) == "tkicker":
                    elem_dict[specific_name_insert] = {
                        "S (m)": twiss_table.iloc[i]["S"],
                        "Command": class_map["kicker"],
                        "L (m)": 0,
                        "Hkick (rad)": twiss_table.iloc[i]["HKICK"],
                        "Vkick (rad)": twiss_table.iloc[i]["VKICK"],
                        "Is field error": False,
                        "Field error KNL": [],
                        "Field error KSL": [],
                        "Is ramping": False,
                        "kick ramping file": "",
                    }
                else:
                    print(f"[Get \033[31mTwiss\033[0m From Madx] Warning: we don't support insert {elem_type} type element of {specific_name}")
                    re_match_record = re_match_record.pop()  # delete the last element

        print(f"[Get \033[31mTwiss\033[0m From Madx] Success: {len(re_match_record)} elements have been inserted as thin-lens into elem_dict:")
        print(f"[Get \033[31mTwiss\033[0m From Madx] \t{re_match_record}")

    # ------------------ Read field error ------------------ #

    error_record = []
    if is_field_error:
        error_dict = get_field_error_from_madx_errorfile(error_file_path)

        for key, value in error_dict.items():
            if key in elem_dict.keys():
                elem_dict[f"{key}_error"] = {
                    "S (m)": elem_dict[key]["S (m)"],
                    "Command": class_map["multipole"],
                    "L (m)": 0,
                    "KiL": value["KiL"],
                    "KiSL": value["KiSL"],
                }
                error_record.append(f"{key}_error")
            else:
                print(f"[Get \033[31mTwiss\033[0m From Madx] We don't find {key}[in error file] in provided elem_dict")

        print(f"[Get \033[31mTwiss\033[0m From Madx] Success: {len(error_record)} field errors have been inserted into elem_dict as thin multipole:")
        print(f"[Get \033[31mTwiss\033[0m From Madx] \t{error_record}")

    # ------------------ Check data ------------------ #

    length_count = 0
    for key, value in elem_dict.items():
        if "L (m)" in value.keys():
            l_tmp = value["L (m)"]
            length_count += l_tmp
        else:
            s_tmp = value["S (m)"]
            s_previous_tmp = value["S previous (m)"]
            l_tmp = s_tmp - s_previous_tmp
            length_count += l_tmp
    length_diff = length_count - circumference
    if abs(length_diff) < 1e-6:
        print(
            f"[Get \033[31mTwiss\033[0m From Madx] Pass the circumference test: theory = {circumference} m, current = {length_count} m, diff = {length_diff:.15e} m"
        )
    else:
        print(
            f"[Get \033[31mTwiss\033[0m From Madx] Failed the circumference test: theory = {circumference} m, current = {length_count} m, diff = {length_diff:.15e} m"
        )
        sys.exit(1)

    # ------------------ Finished ------------------ #

    # for key, value in elem_dict.items():
    #     print(f"key: {key}, value: {value}")

    print(
        f"[Get \033[31mTwiss\033[0m From Madx] Success: {len(elem_dict)} elements ({num_elem} twiss + {len(re_match_record)} insert + {len(error_record)} error multipole) have been read from madx twiss file"
    )

    return elem_dict, circumference


def get_twiss_interpolate_from_madx_twissfile(
    twiss_file_path,
    num_interp_slice,
    error_file_path="",
    logi_transfer_method="off",
    muz=0.001,
    DQx=0.0,
    DQy=0.0,
    is_field_error=False,
    insert_element_name_pattern=["BRMG41Q22"],
    interp_kind="cubic",
):
    # ------------------------- Read madx twiss file ------------------------- #

    twiss_table = tfs.read(twiss_file_path)  # TFSDataFrame, which is DataFrame + headers

    headers = twiss_table.headers  # get header information such as particle, energy, etc.
    column_names = twiss_table.columns  # get column names such as NAME, S, BETX, etc.
    shape = twiss_table.shape  # get data shape (rows, columns)

    num_elem = shape[0]
    print(
        f"[Get \033[31mTwiss\033[0m From Madx] There are '{num_elem}' elements in twiss file, the first and last element names are '{twiss_table.iloc[0]['NAME']}' and '{twiss_table.iloc[-1]['NAME']}' respectively."
    )
    circumference = headers["LENGTH"]
    Qx = headers["Q1"]
    Qy = headers["Q2"]
    DQx_file = headers["DQ1"]
    DQy_file = headers["DQ2"]

    if DQx == "from_file":
        DQx = DQx_file
    if DQy == "from_file":
        DQy = DQy_file

    if abs(DQx - DQx_file) > 1e-10:
        print(f"[Get \033[31mTwiss\033[0m From Madx] Warning: DQx in twiss file is {DQx_file}, but the current setting is {DQx}")
    if abs(DQy - DQy_file) > 1e-10:
        print(f"[Get \033[31mTwiss\033[0m From Madx] Warning: DQy in twiss file is {DQy_file}, but the current setting is {DQy}")

    print(f"[Get \033[31mTwiss\033[0m From Madx] Circumference = {circumference}, Qx = {Qx}, Qy ={Qy}, DQx = {DQx}, DQy = {DQy}")

    # ------------------ Generate PASS required twiss data ---------------- #

    elem_dict = {}
    elem_name_list = []
    re_match_record = []
    s_uniform = np.linspace(0, circumference, num_interp_slice, endpoint=True)
    s_insert = []

    # ------------------ Insert element ------------------ #

    if len(insert_element_name_pattern) > 0:
        combined_pattern = re.compile('|'.join(f'({pattern})' for pattern in insert_element_name_pattern))

        for i in np.arange(num_elem):
            elem_name = twiss_table.iloc[i]["NAME"]
            elem_type = twiss_table.iloc[i]["KEYWORD"]

            elem_name_list.append(elem_name)
            elem_count_result = Counter(elem_name_list)
            elem_appear_times = elem_count_result[elem_name]
            specific_name = f"{elem_name}[{elem_appear_times}]"

            is_match = combined_pattern.search(specific_name)

            if is_match:
                specific_name_insert = f"{specific_name}"
                re_match_record.append(specific_name_insert)
                s_insert.append(twiss_table.iloc[i]["S"])

                if elem_type.lower() == "quadrupole":
                    elem_dict[specific_name_insert] = {
                        "S (m)": twiss_table.iloc[i]["S"],
                        "Command": class_map["quadrupole"],
                        "L (m)": 0,
                        "K1L (m^-1)": twiss_table.iloc[i]["K1L"],
                        "K1SL (m^-1)": twiss_table.iloc[i]["K1SL"],
                        "Is field error": False,
                        "Field error KNL": [],
                        "Field error KSL": [],
                        "Is ramping": False,
                        "K1L ramping file": "",
                        "K1SL ramping file": "",
                    }
                elif elem_type.lower() == "sextupole":
                    elem_dict[specific_name_insert] = {
                        "S (m)": twiss_table.iloc[i]["S"],
                        "Command": class_map["sextupole"],
                        "L (m)": 0,
                        "K2L (m^-2)": twiss_table.iloc[i]["K2L"],
                        "K2SL (m^-2)": twiss_table.iloc[i]["K2SL"],
                        "Is field error": False,
                        "Field error KNL": [],
                        "Field error KSL": [],
                        "Is ramping": False,
                        "K2L ramping file": "",
                        "K2SL ramping file": "",
                    }
                elif elem_type.lower() == "octupole":
                    elem_dict[specific_name_insert] = {
                        "S (m)": twiss_table.iloc[i]["S"],
                        "Command": class_map["Octupole"],
                        "L (m)": 0,
                        "K3L (m^-3)": twiss_table.iloc[i]["K3L"],
                        "K3SL (m^-3)": twiss_table.iloc[i]["K3SL"],
                        "Is field error": False,
                        "Field error KNL": [],
                        "Field error KSL": [],
                        "Is ramping": False,
                        "K3L ramping file": "",
                        "K3SL ramping file": "",
                    }
                elif elem_type.lower() == "multipole":
                    elem_dict[specific_name_insert] = {
                        "S (m)": twiss_table.iloc[i]["S"],
                        "Command": class_map["multipole"],
                        "L (m)": 0,
                        "KiL": [],
                        "KiSL": [],
                        "Is ramping": False,
                        "KL ramping file": "",
                    }
                elif elem_type.lower() == "hkicker" or elem_type.lower() == "vkicker" or elem_type.lower() == "kicker" or elem_type.lower(
                ) == "tkicker":
                    elem_dict[specific_name_insert] = {
                        "S (m)": twiss_table.iloc[i]["S"],
                        "Command": class_map["kicker"],
                        "L (m)": 0,
                        "Hkick (rad)": twiss_table.iloc[i]["HKICK"],
                        "Vkick (rad)": twiss_table.iloc[i]["VKICK"],
                        "Is field error": False,
                        "Field error KNL": [],
                        "Field error KSL": [],
                        "Is ramping": False,
                        "kick ramping file": "",
                    }
                else:
                    print(f"[Get \033[31mTwiss\033[0m From Madx] Warning: we don't support insert {elem_type} type element of {specific_name}")
                    re_match_record = re_match_record.pop()  # delete the last element
                    s_insert = s_insert.pop()

        print(f"[Get \033[31mTwiss\033[0m From Madx] Success: {len(re_match_record)} elements have been inserted as thin-lens into elem_dict:")
        print(f"[Get \033[31mTwiss\033[0m From Madx] \t{re_match_record}")

    # ------------------ Read field error ------------------ #

    error_record = []
    if is_field_error:
        error_dict = get_field_error_from_madx_errorfile(error_file_path)

        for key, value in error_dict.items():
            if key in elem_dict.keys():
                elem_dict[key]["Is field error"] = value["isFieldError"]
                elem_dict[key]["Field error KNL"] = value["Field error KNL"]
                elem_dict[key]["Field error KSL"] = value["Field error KSL"]
                error_record.append(f"{key}")
            else:
                print(f"[Get \033[31mTwiss\033[0m From Madx] We don't find {key}[in error file] in provided elem_dict")

        print(f"[Get \033[31mTwiss\033[0m From Madx] Success: {len(error_record)} field errors have been read to existing elements:")
        print(f"[Get \033[31mTwiss\033[0m From Madx] \t{error_record}")

    # ------------------ Interpolate S ------------------ #

    s = twiss_table["S"].to_numpy()
    betx = twiss_table["BETX"].to_numpy()
    bety = twiss_table["BETY"].to_numpy()
    alfx = twiss_table["ALFX"].to_numpy()
    alfy = twiss_table["ALFY"].to_numpy()
    Dx = twiss_table["DX"].to_numpy()
    Dpx = twiss_table["DPX"].to_numpy()
    mux = twiss_table["MUX"].to_numpy()
    muy = twiss_table["MUY"].to_numpy()

    keep_mask = np.ones(len(s), dtype=bool)
    for i in range(1, len(s)):
        if np.abs(s[i] - s[i - 1]) <= 1e-10:
            keep_mask[i] = False

    s = s[keep_mask]  # delete duplicates
    betx = betx[keep_mask]
    bety = bety[keep_mask]
    alfx = alfx[keep_mask]
    alfy = alfy[keep_mask]
    Dx = Dx[keep_mask]
    Dpx = Dpx[keep_mask]
    mux = mux[keep_mask]
    muy = muy[keep_mask]

    interp_func_betx = interpolate.interp1d(s, betx, kind=interp_kind, fill_value="extrapolate")
    interp_func_bety = interpolate.interp1d(s, bety, kind=interp_kind, fill_value="extrapolate")
    interp_func_alfx = interpolate.interp1d(s, alfx, kind=interp_kind, fill_value="extrapolate")
    interp_func_alfy = interpolate.interp1d(s, alfy, kind=interp_kind, fill_value="extrapolate")
    interp_func_Dx = interpolate.interp1d(s, Dx, kind=interp_kind, fill_value="extrapolate")
    interp_func_Dpx = interpolate.interp1d(s, Dpx, kind=interp_kind, fill_value="extrapolate")
    interp_func_mux = interpolate.interp1d(s, mux, kind=interp_kind, fill_value="extrapolate")
    interp_func_muy = interpolate.interp1d(s, muy, kind=interp_kind, fill_value="extrapolate")

    if len(s_insert) == 0:
        s_interp = s_uniform
    else:
        s_insert = np.array(s_insert)
        s_interp = np.concatenate([s_uniform, s_insert])
        s_interp = np.unique(s_interp)  # remove duplicateds and sort

    betx_interp = interp_func_betx(s_interp)
    bety_interp = interp_func_bety(s_interp)
    alfx_interp = interp_func_alfx(s_interp)
    alfy_interp = interp_func_alfy(s_interp)
    Dx_interp = interp_func_Dx(s_interp)
    Dpx_interp = interp_func_Dpx(s_interp)
    mux_interp = interp_func_mux(s_interp)
    muy_interp = interp_func_muy(s_interp)

    # fig, ax = plt.subplots()
    # ax.plot(s, mux)
    # ax.scatter(s_interp, mux_interp)
    # plt.show()

    for i in np.arange(len(s_interp)):
        specific_name = f"Twiss_interp[{i+1}]"

        if i == 0:
            elem_dict[specific_name] = {
                "S (m)": s_interp[i],
                "Command": "Twiss",
                "S previous (m)": s_interp[i],
                "Alpha x": alfx_interp[i],
                "Alpha y": alfy_interp[i],
                "Beta x (m)": betx_interp[i],
                "Beta y (m)": bety_interp[i],
                "Mu x": mux_interp[i],
                "Mu y": muy_interp[i],
                "Mu z": 0.0,
                "Dx (m)": Dx_interp[i],
                "Dpx": Dpx_interp[i],
                "Alpha x previous": alfx_interp[i],
                "Alpha y previous": alfy_interp[i],
                "Beta x previous (m)": betx_interp[i],
                "Beta y previous (m)": bety_interp[i],
                "Mu x previous": mux_interp[i],
                "Mu y previous": muy_interp[i],
                "Mu z previous": 0.0,
                "Dx (m) previous": Dx_interp[i],
                "Dpx previous": Dpx_interp[i],
                "DQx": 0.0,
                "DQy": 0.0,
                "Longitudinal transfer": logi_transfer_method,
            }
        else:
            elem_dict[specific_name] = {
                "S (m)": s_interp[i],
                "Command": "Twiss",
                "S previous (m)": s_interp[i - 1],
                "Alpha x": alfx_interp[i],
                "Alpha y": alfy_interp[i],
                "Beta x (m)": betx_interp[i],
                "Beta y (m)": bety_interp[i],
                "Mu x": mux_interp[i],
                "Mu y": muy_interp[i],
                "Mu z": s_interp[i] / circumference * (muz - 0),
                "Dx (m)": Dx_interp[i],
                "Dpx": Dpx_interp[i],
                "Alpha x previous": alfx_interp[i - 1],
                "Alpha y previous": alfy_interp[i - 1],
                "Beta x previous (m)": betx_interp[i - 1],
                "Beta y previous (m)": bety_interp[i - 1],
                "Mu x previous": mux_interp[i - 1],
                "Mu y previous": muy_interp[i - 1],
                "Mu z previous": s_interp[i - 1] / circumference * (muz - 0),
                "Dx (m) previous": Dx_interp[i - 1],
                "Dpx previous": Dpx_interp[i - 1],
                "DQx": DQx * (mux_interp[i] - mux_interp[i - 1]) / Qx,
                "DQy": DQy * (muy_interp[i] - muy_interp[i - 1]) / Qy,
                # "DQx": DQx * (s[i] - s[i - 1]) / circumference,
                # "DQy": DQy * (s[i] - s[i - 1]) / circumference,
                "Longitudinal transfer": logi_transfer_method,
            }

    print(f"[Get \033[31mTwiss\033[0m From Madx] Success: {len(s_interp)} twiss elements have been interpolated")

    # ------------------ Check data ------------------ #

    length_count = 0
    for key, value in elem_dict.items():
        if "L (m)" in value.keys():
            l_tmp = value["L (m)"]
            length_count += l_tmp
        else:
            s_tmp = value["S (m)"]
            s_previous_tmp = value["S previous (m)"]
            l_tmp = s_tmp - s_previous_tmp
            length_count += l_tmp
    length_diff = length_count - circumference
    if abs(length_diff) < 1e-6:
        print(
            f"[Get \033[31mTwiss\033[0m From Madx] Pass the circumference test: theory = {circumference} m, current = {length_count} m, diff = {length_diff:.15e} m"
        )
    else:
        print(
            f"[Get \033[31mTwiss\033[0m From Madx] Failed the circumference test: theory = {circumference} m, current = {length_count} m, diff = {length_diff:.15e} m"
        )
        sys.exit(1)

    # ------------------ Finished ------------------ #

    # for key, value in elem_dict.items():
    #     print(f"key: {key}, value: {value}")

    print(
        f"[Get \033[31mTwiss\033[0m From Madx] Success: {len(elem_dict)} ({len(s_interp)} interp + {len(re_match_record)} insert) elements have been read from madx twiss file"
    )

    return elem_dict, circumference


if __name__ == "__main__":
    # elem_dict, circum = get_twiss_from_madx_twissfile(
    #     twiss_file_path=r"C:\Users\changmx\Documents\PASS\para\bring.tfs",
    #     error_file_path=r"C:\Users\changmx\Documents\PASS\para\error_sextupoleerror.tfs",
    #     logi_transfer_method="off",
    #     muz=0.001,
    #     DQx=0.0,
    #     DQy=0.0,
    #     is_field_error=False,
    #     insert_element_name_pattern=["BRMG41Q22"],
    # )

    elem_dict, circum = get_twiss_interpolate_from_madx_twissfile(
        twiss_file_path=r"C:\Users\changmx\Documents\PASS\para\bring.tfs",
        num_interp_slice=100,
        error_file_path=r"C:\Users\changmx\Documents\PASS\para\error_sextupoleerror.tfs",
        logi_transfer_method="off",
        muz=0.001,
        DQx=0.0,
        DQy=0.0,
        is_field_error=False,
        insert_element_name_pattern=["BRMG41Q22"],
        interp_kind="cubic",
    )
    # print(elem_dict)
