import numpy
import random
import json
import re
import os
import sys
import pandas as pd

from cpymad.madx import Madx


def convert_string(s):
    """
    将字符串转换为指定格式：
    - "rb" → "rb:1"
    - "rb[2]" → "rb:2"
    - 处理复杂情况如 "abc123[99]", "x.y_z[3]"

    参数:
    s -- 输入字符串

    返回:
    转换后的字符串
    """
    # 匹配以数字结尾的方括号模式
    pattern = r"\[(\d+)\]$"
    match = re.search(pattern, s)

    if match:
        # 提取数字部分
        num = match.group(1)
        # 移除方括号部分
        base = re.sub(pattern, "", s)
        return f"{base}:{num}"
    else:
        # 没有方括号，直接添加 :1
        return f"{s}:1"


def get_element_from_seq(
    seq_file,
    seq_name,
    is_beam_in_seq=False,
    particle="proton",
    energy=1000,
):
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

    elem_dict = {}

    for i in range(num_elem):
        # print(seq[i]._attr)   # show all elements attributes
        node_name = seq[i].node_name
        name = convert_string(node_name)

        if name in elem_dict:
            print(f"Error: element '{name}' is already exists in element dict")
            sys.exit(1)

        elem_dict[name] = {
            "s": seq[i].at,
            "length": seq[i].length,
            "type": seq[i].base_name,
        }

    # print(elem_dict)
    
    madx.quit()
    return elem_dict


def get_error(
    seq_file,
    error_file,
    seq_name="ring",
    is_beam_in_seq=False,
    particle="proton",
    energy=1000,
):
    madx = Madx()
    madx.option(echo=False)

    madx.call(file=seq_file)
    if not is_beam_in_seq:
        madx.command.beam(sequence=seq_name, particle=particle, energy=energy)
    madx.use(sequence=seq_name)

    madx.call(file=error_file)

    madx.command.select(flag="error", full=True)
    madx.command.etable(table="myerrortable")
    error = madx.table["myerrortable"]
    error_df = error.dframe()
    # error_df.to_csv(r"D:\PASS\para\error_df.csv", index=False)
    # print(error_df)
    nrow_error, ncol_error = error_df.shape
    # print(f"Size of error file '{error_file}' = {nrow_error}")

    elem_dict = get_element_from_seq(seq_file, seq_name)
    error_dict = {}

    epsilon = 1e-10

    for index, row in error_df.iterrows():
        max_order = -1 # -1 means no error. 0 refers to dipole field error, 1 refers to quad. field error , etc.

        name = row["name"]
        for i in range(0, 21):
            col_name = f"k{i}l"
            kil = row[col_name]
            if abs(kil) > epsilon:
                max_order = max(max_order, i)
                # print(name, kil)
        for i in range(0, 21):
            col_name = f"k{i}sl"
            kisl = row[col_name]
            if abs(kisl) > epsilon:
                max_order = max(max_order, i)
                # print(name, kil)
        if max_order > -1:
            knl_list = []
            ksl_list = []

            for i in range(0, max_order + 1):
                col_name_knl = f"k{i}l"
                col_name_ksl = f"k{i}sl"
                knl_list.append(row[col_name_knl])
                ksl_list.append(row[col_name_ksl])

            s = elem_dict[name]["s"]
            length = elem_dict[name]["length"]
            elem_type = elem_dict[name]["type"]

            if name in error_dict:
                print(f"Error: error '{name}' is already exists in element dict")
                sys.exit(1)

            error_dict[name] = {
                "s": s,
                "length": length,
                "type": elem_type,
                "errorOrder": max_order,
                "knl": knl_list,
                "ksl": ksl_list,
            }

        # print(f"name = {name}, k0l = {k0l}, k1l = {k1l}, k2l = {k2l}")
        # print(f"name = {name}, s = {s}, length = {length}")

    # for key, sub_dict in error_dict.items():
    #     print(f"\n{key}:")
    #     print(sub_dict)
    madx.quit()
    return error_dict


if __name__ == "__main__":

    # get_element_info_from_seq(
    #     seq_file=r"D:\PASS\para\BRING2021_03_02.seq",
    #     seq_name="RING",
    # )

    get_error(
        seq_file=r"D:\PASS\para\BRING2021_03_02.seq",
        error_file=r"D:\PASS\para\error.madx",
        seq_name="RING",
    )
