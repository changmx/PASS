"""Read MADX twiss TFS file → PASS Element schema objects.

Unlike madx_twiss (which creates TwissPoint transport points),
this reader creates actual physical elements (Drift, SBend, Quadrupole, etc.)
with their full parameters, suitable for element-by-element tracking.
"""

import numpy as np
import sys
from collections import Counter
import re
import tfs

from PASS.para.schema.elements import (
    DriftElement, MarkerElement, SBendElement, QuadrupoleElement,
    SextupoleElement, OctupoleElement, MultipoleElement, KickerElement,
)
from PASS.para.readers.madx_error import read_madx_errors  # noqa: F401 (re-export alias)
from PASS.para.toolkit import class_map


def merge_drift_elements(items: list, names: list[str]) -> tuple[list, list[str]]:
    """Merge consecutive DriftElements into one.

    Returns (merged_items, merged_names).
    """
    if not items:
        return [], []

    result_items = []
    result_names = []
    i = 0

    while i < len(items):
        current = items[i]
        current_cmd = current.command

        if current_cmd != "Drift":
            result_items.append(current)
            result_names.append(names[i])
            i += 1
            continue

        # Collect consecutive drifts
        drift_indices = [i]
        drift_len = current.length
        s_val = current.s

        j = i + 1
        while j < len(items):
            if items[j].command == "Drift":
                drift_indices.append(j)
                drift_len += items[j].length
                j += 1
            else:
                break

        if len(drift_indices) == 1:
            result_items.append(current)
            result_names.append(names[i])
        else:
            merged = DriftElement(s=s_val, length=drift_len)
            merged_name = "_".join(names[k] for k in drift_indices)
            result_items.append(merged)
            result_names.append(merged_name)

        i = j

    print(f"[Read MADX Elements] Merged drifts: {len(items)} → {len(result_items)}")
    return result_items, result_names


def read_madx_elements(
    twiss_file: str,
    error_file: str = "",
    is_merge_drift: bool = False,
    is_field_error: bool = False,
) -> tuple[list, float]:
    """Read a MADX twiss TFS file → (element_items, circumference).

    Each MADX element is converted to the corresponding PASS Element schema
    object with its physical parameters (length, strength, edge angles, etc.).

    Args:
        twiss_file: path to MADX twiss TFS file.
        error_file: path to MADX error TFS file.
        is_merge_drift: merge consecutive drift elements.
        is_field_error: attach field errors to matching elements.

    Returns:
        (items, circumference) where items is a list of Element schema objects.
    """
    twiss_table = tfs.read(twiss_file)
    headers = twiss_table.headers
    num_elem = twiss_table.shape[0]

    circumference = headers["LENGTH"]
    print(f"[Read MADX Elements] {num_elem} elements, C={circumference}")

    items = []
    names = []
    name_list = []

    for i in range(num_elem):
        elem_name = twiss_table.iloc[i]["NAME"]
        elem_type = twiss_table.iloc[i]["KEYWORD"]
        s = twiss_table.iloc[i]["S"]
        l = twiss_table.iloc[i]["L"]

        name_list.append(elem_name)
        count = Counter(name_list)
        occurrence = count[elem_name]
        specific_name = f"{elem_name}[{occurrence}]"

        et = elem_type.lower()

        if et == "marker":
            item = MarkerElement(s=s)
        elif et == "drift":
            item = DriftElement(s=s, length=l)
        elif et in ("sbend", "rbend"):
            fint = twiss_table.iloc[i]["FINT"]
            fintx = twiss_table.iloc[i]["FINTX"]
            if fintx <= 0:
                fintx = fint
            item = SBendElement(
                s=s, length=l,
                k0l=twiss_table.iloc[i]["ANGLE"],
                e1=twiss_table.iloc[i]["E1"],
                e2=twiss_table.iloc[i]["E2"],
                hgap=twiss_table.iloc[i]["HGAP"],
                fint=fint, fintx=fintx,
            )
        elif et == "quadrupole":
            item = QuadrupoleElement(
                s=s, length=l,
                k1l=twiss_table.iloc[i]["K1L"],
                k1sl=twiss_table.iloc[i]["K1SL"],
            )
        elif et == "sextupole":
            item = SextupoleElement(
                s=s, length=l,
                k2l=twiss_table.iloc[i]["K2L"],
                k2sl=twiss_table.iloc[i]["K2SL"],
            )
        elif et == "octupole":
            item = OctupoleElement(
                s=s, length=l,
                k3l=twiss_table.iloc[i]["K3L"],
                k3sl=twiss_table.iloc[i]["K3SL"],
            )
        elif et == "multipole":
            item = MultipoleElement(s=s, length=l, knl=[], ksl=[])
        elif et in ("hkicker", "vkicker", "kicker", "tkicker"):
            item = KickerElement(
                s=s, length=l,
                hkick=twiss_table.iloc[i]["HKICK"],
                vkick=twiss_table.iloc[i]["VKICK"],
            )
        elif et == "monitor":
            item = DriftElement(s=s, length=l)
        else:
            print(f"[Read MADX Elements] Warning: unsupported {et} '{specific_name}' → drift")
            item = DriftElement(s=s, length=l)

        items.append(item)
        names.append(specific_name)

    # Merge drifts
    if is_merge_drift:
        items, names = merge_drift_elements(items, names)

    # Field errors
    if is_field_error and error_file:
        error_dict = read_madx_errors(error_file)
        name_to_idx = {n: idx for idx, n in enumerate(names)}
        error_count = 0
        for key, errs in error_dict.items():
            if key in name_to_idx:
                idx = name_to_idx[key]
                # Attach error to existing element
                items[idx].is_field_error = True
                items[idx].field_error_knl = errs["knl"]
                items[idx].field_error_ksl = errs["ksl"]
                error_count += 1
            else:
                print(f"[Read MADX Elements] Warning: error '{key}' not found")
        print(f"[Read MADX Elements] {error_count} field errors attached")

    # Circumference check
    length_count = sum(item.length for item in items)
    diff = length_count - circumference
    if abs(diff) < 1e-6:
        print(f"[Read MADX Elements] Circumference check passed: {length_count:.6f} m")
    else:
        print(f"[Read MADX Elements] Circumference check FAILED: "
              f"theory={circumference}, actual={length_count}, diff={diff:.6e}")

    return items, circumference
