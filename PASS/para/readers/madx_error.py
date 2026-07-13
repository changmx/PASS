"""Read field errors from a MADX error TFS file.

Returns a dict: {element_name[n]: {"knl": [...], "ksl": [...]}}
where element_name uses the same naming convention as the twiss reader
(name[occurrence]) so errors can be matched to twiss/element entries.
"""

import numpy as np
from collections import Counter
import tfs


def read_madx_errors(error_file_path: str) -> dict[str, dict]:
    """Read field errors from a MADX error TFS file.

    The error file must contain columns K0L, K1L, ..., K20L and
    K0SL, K1SL, ..., K20SL (standard MADX EFCOMP output).

    Args:
        error_file_path: path to the MADX error TFS file.

    Returns:
        {specific_name: {"knl": [k0l, k1l, ...], "ksl": [k0sl, k1sl, ...]}}
        where specific_name = "NAME[occurrence]".
        Only elements with non-zero errors are included.
    """
    error_table = tfs.read(error_file_path)
    num_elem = error_table.shape[0]

    print(
        f"[Read MADX Errors] {num_elem} elements in error file, "
        f"first='{error_table.iloc[0]['NAME']}', last='{error_table.iloc[-1]['NAME']}'"
    )

    error_dict = {}
    elem_name_list = []

    for i in range(num_elem):
        elem_name = error_table.iloc[i]["NAME"]

        elem_name_list.append(elem_name)
        count = Counter(elem_name_list)
        occurrence = count[elem_name]
        specific_name = f"{elem_name}[{occurrence}]"

        # Find max order with non-zero error
        max_order = -1
        for iorder in range(0, 21):
            kil = error_table.iloc[i][f"K{iorder}L"]
            if abs(kil) > 1e-10:
                max_order = max(max_order, iorder)
        for iorder in range(0, 21):
            kisl = error_table.iloc[i][f"K{iorder}SL"]
            if abs(kisl) > 1e-10:
                max_order = max(max_order, iorder)

        if max_order > -1:
            knl = []
            ksl = []
            for iorder in range(0, max_order + 1):
                knl.append(error_table.iloc[i][f"K{iorder}L"])
                ksl.append(error_table.iloc[i][f"K{iorder}SL"])

            error_dict[specific_name] = {"knl": knl, "ksl": ksl}

    print(f"[Read MADX Errors] {len(error_dict)} elements with non-zero errors")
    return error_dict
