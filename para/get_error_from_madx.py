import numpy
import random
import json
import re
import os
import sys
import pandas as pd
from collections import Counter

import tfs  # tfs_pandas, https://pylhc.github.io/tfs/


def get_field_error_from_madx_errorfile(error_file_path):

    # ------------------------- Read madx error output ------------------------- #
    error_table = tfs.read(error_file_path)  # TFSDataFrame, which is DataFrame + headers

    headers = error_table.headers
    column_names = error_table.columns  # get column names such as NAME, S, BETX, etc.
    shape = error_table.shape  # get data shape (rows, columns)

    num_elem = shape[0]
    print(
        f"[Get \033[32mError\033[0m From Madx] There are '{num_elem}' error elements in error file, the first and last element names are '{error_table.iloc[0]['NAME']}' and '{error_table.iloc[-1]['NAME']}' respectively."
    )
    # print(error_table)

    # ------------------------- Add error data to elem_dict ------------------------- #

    error_dict = {}
    elem_name_list = []

    for i in range(num_elem):
        elem_name = error_table.iloc[i]["NAME"]

        elem_name_list.append(elem_name)
        elem_count_result = Counter(elem_name_list)
        elem_appear_times = elem_count_result[elem_name]
        specific_name = f"{elem_name}[{elem_appear_times}]"

        max_order = -1  # -1 means no error. 0 refers to dipole field error, 1 refers to quad. field error , etc.

        for iorder in range(0, 21):
            kil = error_table.iloc[i][f"K{iorder}L"]
            if abs(kil) > 1e-10:
                max_order = max(max_order, iorder)
        for iorder in range(0, 21):
            kisl = error_table.iloc[i][f"K{iorder}SL"]
            if abs(kisl) > 1e-10:
                max_order = max(max_order, iorder)

        if max_order > -1:
            kl_list = []
            ksl_list = []

            for iorder in range(0, max_order + 1):
                kl_list.append(error_table.iloc[i][f"K{iorder}L"])
                ksl_list.append(error_table.iloc[i][f"K{iorder}SL"])

            error_dict[specific_name] = {
                "Is field error": True,
                "Field error KNL": kl_list,
                "Field error KSL": ksl_list,
            }

    return error_dict


if __name__ == "__main__":
    pass
