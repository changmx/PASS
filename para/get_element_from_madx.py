import numpy as np
import sys
from collections import Counter
import re

import tfs  # tfs_pandas, https://pylhc.github.io/tfs/
from get_error_from_madx import get_field_error_from_madx_errorfile
from toolkit import class_map


def merge_drift_element(elem_dict):
    if not elem_dict:
        return {}

    # 获取所有键并保持顺序
    keys = list(elem_dict.keys())
    result = dict()
    i = 0

    while i < len(keys):
        current_key = keys[i]
        current_value = elem_dict[current_key]

        # 如果不是DriftElement，直接添加到结果
        if current_value.get('Command') != 'DriftElement':
            result[current_key] = current_value
            i += 1
            continue

        # 如果是DriftElement，查找相邻的DriftElement
        drift_keys = [current_key]
        drift_L_sum = current_value['L (m)']
        s_value = current_value['S (m)']

        j = i + 1
        while j < len(keys):
            next_key = keys[j]
            next_value = elem_dict[next_key]

            if next_value.get('Command') == 'DriftElement':
                drift_keys.append(next_key)
                drift_L_sum += next_value['L (m)']
                j += 1
            else:
                break

        # 如果只有一个DriftElement，直接使用原键
        if len(drift_keys) == 1:
            result[current_key] = current_value
        else:
            # 合并多个DriftElement
            merged_key = '_'.join(drift_keys)
            merged_value = {'S (m)': s_value, 'Command': 'DriftElement', 'L (m)': drift_L_sum}
            result[merged_key] = merged_value

        i = j  # 跳过已处理的元素

    print(
        f"[Get \033[33mElement\033[0m From Madx] Merge drift elements successfully. Number of original elements is: {len(elem_dict)}, after merging is: {len(result)}"
    )
    return dict(result)


def get_element_from_madx_twissfile(
    twiss_file_path,
    error_file_path,
    is_merge_drift=False,
    is_field_error=False,
):

    # ------------------------- Read madx twiss file ------------------------- #

    twiss_table = tfs.read(twiss_file_path)  # TFSDataFrame, which is DataFrame + headers

    headers = twiss_table.headers  # get header information such as particle, energy, etc.
    column_names = twiss_table.columns  # get column names such as NAME, S, BETX, etc.
    shape = twiss_table.shape  # get data shape (rows, columns)

    num_elem = shape[0]
    print(
        f"[Get \033[33mElement\033[0m From Madx] There are '{num_elem}' elements in twiss file, the first and last element names are '{twiss_table.iloc[0]['NAME']}' and '{twiss_table.iloc[-1]['NAME']}' respectively."
    )
    circumference = headers["LENGTH"]
    Qx = headers["Q1"]
    Qy = headers["Q2"]
    DQx_file = headers["DQ1"]
    DQy_file = headers["DQ2"]

    print(f"[Get \033[33mElement\033[0m From Madx] Circumference = {circumference}, Qx = {Qx}, Qy ={Qy}, DQx = {DQx_file}, DQy = {DQy_file}")

    # ------------------ Generate PASS required element data ---------------- #

    elem_dict = {}
    elem_name_list = []

    for i in range(num_elem):
        elem_name = twiss_table.iloc[i]["NAME"]
        elem_type = twiss_table.iloc[i]["KEYWORD"]
        s = twiss_table.iloc[i]["S"]
        l = twiss_table.iloc[i]["L"]

        elem_name_list.append(elem_name)
        elem_count_result = Counter(elem_name_list)
        elem_appear_times = elem_count_result[elem_name]
        specific_name = f"{elem_name}[{elem_appear_times}]"

        if elem_type.lower() == "marker":
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["marker"],
            }
        elif elem_type.lower() == "drift":
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["drift"],
                "L (m)": l,
            }
        elif elem_type.lower() == "sbend":
            fint = twiss_table.iloc[i]["FINT"]
            fintx = twiss_table.iloc[i]["FINTX"]
            if fintx <= 0:
                fintx = fint
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["sbend"],
                "L (m)": l,
                "Angle (rad)": twiss_table.iloc[i]["ANGLE"],
                "E1 (rad)": twiss_table.iloc[i]["E1"],
                "E2 (rad)": twiss_table.iloc[i]["E2"],
                "Hgap (m)": twiss_table.iloc[i]["HGAP"],
                "Fint": fint,
                "Fintx": fintx,
                "Is field error": False,
                "Field error KNL": [],
                "Field error KSL": [],
                "Is ramping": False,
                "K0L ramping file": "",
            }
        elif elem_type.lower() == "rbend":
            fint = twiss_table.iloc[i]["FINT"]
            fintx = twiss_table.iloc[i]["FINTX"]
            if fintx <= 0:
                fintx = fint
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["rbend"],
                "L (m)": l,
                "Angle (rad)": twiss_table.iloc[i]["ANGLE"],
                "E1 (rad)": twiss_table.iloc[i]["E1"],
                "E2 (rad)": twiss_table.iloc[i]["E2"],
                "Hgap (m)": twiss_table.iloc[i]["HGAP"],
                "Fint": fint,
                "Fintx": fintx,
                "Is field error": False,
                "Field error KNL": [],
                "Field error KSL": [],
                "Is ramping": False,
                "K0L ramping file": "",
            }
        elif elem_type.lower() == "quadrupole":
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["quadrupole"],
                "L (m)": l,
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
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["sextupole"],
                "L (m)": l,
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
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["Octupole"],
                "L (m)": l,
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
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["multipole"],
                "L (m)": l,
                "KiL": [],
                "KiSL": [],
                "Is ramping": False,
                "KL ramping file": "",
            }
        elif elem_type.lower() == "hkicker" or elem_type.lower() == "vkicker" or elem_type.lower() == "kicker" or elem_type.lower() == "tkicker":
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["kicker"],
                "L (m)": l,
                "Hkick (rad)": twiss_table.iloc[i]["HKICK"],
                "Vkick (rad)": twiss_table.iloc[i]["VKICK"],
                "Is field error": False,
                "Field error KNL": [],
                "Field error KSL": [],
                "Is ramping": False,
                "kick ramping file": "",
            }
        elif elem_type.lower() == "monitor":
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["drift"],
                "L (m)": l,
            }
        else:
            print(
                f"[Get \033[33mElement\033[0m From Madx] Warning: we don't support {elem_type} ({elem_name}) @ S={s} now. This element is treated as drift now."
            )
            elem_dict[specific_name] = {
                "S (m)": s,
                "Command": class_map["drift"],
                "L (m)": l,
            }

    # ------------------ Merge drift ------------------ #
    if is_merge_drift:
        elem_dict = merge_drift_element(elem_dict)

    # ------------------ Read field error ------------------ #

    if is_field_error:
        error_record = []
        error_dict = get_field_error_from_madx_errorfile(error_file_path)

        for key, value in error_dict.items():
            if key in elem_dict.keys():
                elem_dict[key]["Is field error"] = value["Is field error"]
                elem_dict[key]["Field error KNL"] = value["Field error KNL"]
                elem_dict[key]["Field error KSL"] = value["Field error KSL"]

                error_record.append(key)
            else:
                print(f"[Get \033[33mElement\033[0m From Madx] We don't find {key}[in error file] in provided elem_dict")

        print(f"[Get \033[33mElement\033[0m From Madx] Success: {len(error_record)} field errors have been read to existing elements:")
        print(f"[Get \033[33mElement\033[0m From Madx] \t{error_record}")

    # ------------------ Check data ------------------ #

    length_count = 0
    for key, value in elem_dict.items():
        if "L (m)" in value.keys():
            l = value["L (m)"]
            length_count += l
    length_diff = length_count - circumference
    if abs(length_diff) < 1e-6:
        print(
            f"[Get \033[33mElement\033[0m From Madx] Pass the circumference test: theory = {circumference} m, current = {length_count} m, diff = {length_diff:.15e} m"
        )
    else:
        print(
            f"[Get \033[33mElement\033[0m From Madx] Failed the circumference test: theory = {circumference} m, current = {length_count} m, diff = {length_diff:.15e} m"
        )
        sys.exit(1)

    # ------------------ Finished ------------------ #

    # for key, value in elem_dict.items():
    #     if "L (m)" in value.keys():
    #         # print(f"{key}, S = {value["S (m)"]}, L = {value["L (m)"]}")
    #         print(f"key: {key}, value: {value}")

    print(f"[Get \033[33mElement\033[0m From Madx] Success: {len(elem_dict)} elements have been read from madx twiss file")

    return elem_dict, circumference


def add_ramping_file(sequence=dict(), elem_name_re_pattern=[], file_key="", file_path=""):

    if len(elem_name_re_pattern) == 0:
        return

    combined_pattern = re.compile('|'.join(f'({pattern})' for pattern in elem_name_re_pattern))

    ramping_record = []
    for key, value in sequence.items():
        is_match = combined_pattern.search(key)
        if is_match and ("Element" in value["Command"]):
            if file_key in value:
                value[file_key] = file_path
                ramping_record.append(key)
            else:
                print(f"[Add Ramping File] We don't find key '{file_key}' in '{key}'")

    print(f"[Add Ramping File] We have add ramping file '{file_path}' to elements: {ramping_record}")


if __name__ == "__main__":

    elem_dict, circumference = get_element_from_madx_twissfile(
        twiss_file_path=r"C:\Users\changmx\Documents\PASS\para\bring.tfs",
        error_file_path=r"C:\Users\changmx\Documents\PASS\para\error_sextupoleerror.tfs",
        is_merge_drift=True,
        is_field_error=True,
    )
