"""Read MADX twiss TFS file → PASS TwissPoint schema objects.

Two modes:
    1. Direct: read every element in the twiss file as a TwissPoint
    2. Interpolate: resample twiss parameters onto a uniform s-grid

Also supports inserting specific elements (quad/sext/oct/kicker/multipole)
as thin-lens elements alongside the twiss points.

The returned list can be fed directly into a Sequence container.
"""

import numpy as np
from collections import Counter
from scipy import interpolate
import tfs

from PASS.para.schema.twiss import TwissPoint
from PASS.para.schema.elements import (
    QuadrupoleElement, SextupoleElement, OctupoleElement,
    MultipoleElement, KickerElement,
)
from PASS.para.readers.madx_error import read_madx_errors
from PASS.para.toolkit import class_map


def read_madx_twiss(
    twiss_file: str,
    error_file: str = "",
    muz: float = 0.0,
    dqx: float | str = "from_file",
    dqy: float | str = "from_file",
    is_field_error: bool = False,
    insert_patterns: list[str] | None = None,
    longitudinal_transfer: str = "off",
) -> tuple[list, float]:
    """Read a MADX twiss TFS file → (schema_items, circumference).

    Each row in the twiss file becomes a TwissPoint with current + previous
    optical functions. The first point has previous = current.

    Args:
        twiss_file: path to the MADX twiss TFS file.
        error_file: path to the MADX error TFS file (for field errors).
        muz: longitudinal tune (default 0.0).
        dqx: chromaticity Qx. Float or "from_file" to read from headers.
        dqy: chromaticity Qy. Float or "from_file" to read from headers.
        is_field_error: if True, read field errors and attach as multipole elements.
        insert_patterns: regex patterns to match element names for thin-lens insertion.
        longitudinal_transfer: "off" / "drift" / "matrix".

    Returns:
        (items, circumference) where items is a list of TwissPoint and
        optionally Element objects (if insert_patterns or is_field_error).
        Also returns a dict mapping item names for Sequence assembly.
    """
    twiss_table = tfs.read(twiss_file)
    headers = twiss_table.headers
    num_elem = twiss_table.shape[0]

    circumference = headers["LENGTH"]
    qx = headers["Q1"]
    qy = headers["Q2"]
    dqx_file = headers["DQ1"]
    dqy_file = headers["DQ2"]

    if dqx == "from_file":
        dqx = dqx_file
    if dqy == "from_file":
        dqy = dqy_file

    if abs(dqx - dqx_file) > 1e-10:
        print(f"[Read MADX Twiss] Warning: DQx file={dqx_file}, setting={dqx}")
    if abs(dqy - dqy_file) > 1e-10:
        print(f"[Read MADX Twiss] Warning: DQy file={dqy_file}, setting={dqy}")

    print(f"[Read MADX Twiss] {num_elem} elements, C={circumference}, "
          f"Qx={qx}, Qy={qy}, DQx={dqx}, DQy={dqy}")

    betx = twiss_table["BETX"]
    bety = twiss_table["BETY"]
    alfx = twiss_table["ALFX"]
    alfy = twiss_table["ALFY"]
    dx = twiss_table["DX"]
    dpx = twiss_table["DPX"]
    mux = twiss_table["MUX"]
    muy = twiss_table["MUY"]
    s = twiss_table["S"]

    items = []
    name_map = {}  # name → index in items
    elem_name_list = []

    for i in range(num_elem):
        elem_name = twiss_table.iloc[i]["NAME"]
        elem_name_list.append(elem_name)
        count = Counter(elem_name_list)
        occurrence = count[elem_name]
        specific_name = f"{elem_name}[{occurrence}]"

        if i == 0:
            tp = TwissPoint(
                s=s[i], s_previous=s[i],
                alpha_x=alfx[i], alpha_y=alfy[i],
                beta_x=betx[i], beta_y=bety[i],
                mu_x=mux[i], mu_y=muy[i], mu_z=0.0,
                dx=dx[i], dpx=dpx[i],
                alpha_x_previous=alfx[i], alpha_y_previous=alfy[i],
                beta_x_previous=betx[i], beta_y_previous=bety[i],
                mu_x_previous=mux[i], mu_y_previous=muy[i], mu_z_previous=0.0,
                dx_previous=dx[i], dpx_previous=dpx[i],
                dqx=0.0, dqy=0.0,
                longitudinal_transfer=longitudinal_transfer,
            )
        else:
            mu_z_i = s[i] / circumference * muz
            mu_z_prev = s[i - 1] / circumference * muz
            tp = TwissPoint(
                s=s[i], s_previous=s[i - 1],
                alpha_x=alfx[i], alpha_y=alfy[i],
                beta_x=betx[i], beta_y=bety[i],
                mu_x=mux[i], mu_y=muy[i], mu_z=mu_z_i,
                dx=dx[i], dpx=dpx[i],
                alpha_x_previous=alfx[i - 1], alpha_y_previous=alfy[i - 1],
                beta_x_previous=betx[i - 1], beta_y_previous=bety[i - 1],
                mu_x_previous=mux[i - 1], mu_y_previous=muy[i - 1], mu_z_previous=mu_z_prev,
                dx_previous=dx[i - 1], dpx_previous=dpx[i - 1],
                dqx=dqx * (mux[i] - mux[i - 1]) / qx,
                dqy=dqy * (muy[i] - muy[i - 1]) / qy,
                longitudinal_transfer=longitudinal_transfer,
            )
        items.append(tp)
        name_map[specific_name] = len(items) - 1

    print(f"[Read MADX Twiss] {len(items)} twiss points created")

    # --- Insert thin-lens elements ---
    if insert_patterns:
        insert_items, insert_names = _insert_elements(
            twiss_table, insert_patterns, circumference
        )
        items.extend(insert_items)
        for name, idx_offset in zip(insert_names, range(len(items) - len(insert_items), len(items))):
            name_map[name] = idx_offset
        print(f"[Read MADX Twiss] {len(insert_items)} thin-lens elements inserted")

    # --- Attach field errors ---
    if is_field_error and error_file:
        error_dict = read_madx_errors(error_file)
        error_items = []
        error_names = []
        for key, errs in error_dict.items():
            if key in name_map:
                idx = name_map[key]
                s_val = items[idx].s if hasattr(items[idx], "s") else items[idx].model_dump(by_alias=True)["S (m)"]
                err_item = MultipoleElement(
                    s=s_val, length=0.0,
                    knl=errs["knl"], ksl=errs["ksl"],
                )
                error_items.append(err_item)
                error_names.append(f"{key}_error")
            else:
                print(f"[Read MADX Twiss] Warning: error element '{key}' not found in twiss")
        items.extend(error_items)
        for name in error_names:
            name_map[name] = len(items) - len(error_names) + error_names.index(name)
        print(f"[Read MADX Twiss] {len(error_items)} field error multipoles added")

    # --- Circumference check ---
    length_count = 0.0
    for item in items:
        d = item.model_dump(by_alias=True)
        if "L (m)" in d and d["L (m)"] > 0:
            length_count += d["L (m)"]
        elif "S previous (m)" in d:
            length_count += d["S (m)"] - d["S previous (m)"]

    diff = length_count - circumference
    if abs(diff) < 1e-6:
        print(f"[Read MADX Twiss] Circumference check passed: {length_count:.6f} m")
    else:
        print(f"[Read MADX Twiss] Circumference check FAILED: "
              f"theory={circumference}, actual={length_count}, diff={diff:.6e}")

    return items, circumference


def _insert_elements(
    twiss_table, insert_patterns: list[str], circumference: float
) -> tuple[list, list[str]]:
    """Create thin-lens elements for names matching insert_patterns."""
    import re

    combined = re.compile("|".join(f"({p})" for p in insert_patterns))
    items = []
    names = []
    name_list = []

    for i in range(len(twiss_table)):
        elem_name = twiss_table.iloc[i]["NAME"]
        elem_type = twiss_table.iloc[i]["KEYWORD"]

        name_list.append(elem_name)
        count = Counter(name_list)
        occurrence = count[elem_name]
        specific_name = f"{elem_name}[{occurrence}]"

        if not combined.search(specific_name):
            continue

        s = twiss_table.iloc[i]["S"]
        et = elem_type.lower()

        if et == "quadrupole":
            item = QuadrupoleElement(
                s=s, length=0.0,
                k1l=twiss_table.iloc[i]["K1L"],
                k1sl=twiss_table.iloc[i]["K1SL"],
            )
        elif et == "sextupole":
            item = SextupoleElement(
                s=s, length=0.0,
                k2l=twiss_table.iloc[i]["K2L"],
                k2sl=twiss_table.iloc[i]["K2SL"],
            )
        elif et == "octupole":
            item = OctupoleElement(
                s=s, length=0.0,
                k3l=twiss_table.iloc[i]["K3L"],
                k3sl=twiss_table.iloc[i]["K3SL"],
            )
        elif et == "multipole":
            item = MultipoleElement(s=s, length=0.0, knl=[], ksl=[])
        elif et in ("hkicker", "vkicker", "kicker", "tkicker"):
            item = KickerElement(
                s=s, length=0.0,
                hkick=twiss_table.iloc[i]["HKICK"],
                vkick=twiss_table.iloc[i]["VKICK"],
            )
        else:
            print(f"[Read MADX Twiss] Warning: cannot insert {et} '{specific_name}', skipping")
            continue

        items.append(item)
        names.append(specific_name)

    return items, names


def read_madx_twiss_interpolated(
    twiss_file: str,
    num_interp_slice: int,
    error_file: str = "",
    muz: float = 0.001,
    dqx: float = 0.0,
    dqy: float = 0.0,
    is_field_error: bool = False,
    insert_patterns: list[str] | None = None,
    longitudinal_transfer: str = "off",
    interp_kind: str = "cubic",
) -> tuple[list, float]:
    """Read MADX twiss and interpolate onto a uniform s-grid.

    Produces num_interp_slice TwissPoints evenly spaced along the ring.
    Insert elements (if any) are placed at their original s positions.
    """
    twiss_table = tfs.read(twiss_file)
    headers = twiss_table.headers
    num_elem = twiss_table.shape[0]

    circumference = headers["LENGTH"]
    qx = headers["Q1"]
    qy = headers["Q2"]

    if dqx == "from_file":
        dqx = headers["DQ1"]
    if dqy == "from_file":
        dqy = headers["DQ2"]

    print(f"[Read MADX Twiss] Interpolated: {num_interp_slice} slices, "
          f"C={circumference}, Qx={qx}, Qy={qy}")

    # Read + clean raw data
    s = twiss_table["S"].to_numpy()
    betx = twiss_table["BETX"].to_numpy()
    bety = twiss_table["BETY"].to_numpy()
    alfx = twiss_table["ALFX"].to_numpy()
    alfy = twiss_table["ALFY"].to_numpy()
    dx = twiss_table["DX"].to_numpy()
    dpx = twiss_table["DPX"].to_numpy()
    mux = twiss_table["MUX"].to_numpy()
    muy = twiss_table["MUY"].to_numpy()

    # Remove duplicate s-points
    keep_mask = np.ones(len(s), dtype=bool)
    for i in range(1, len(s)):
        if abs(s[i] - s[i - 1]) <= 1e-10:
            keep_mask[i] = False
    s, betx, bety = s[keep_mask], betx[keep_mask], bety[keep_mask]
    alfx, alfy = alfx[keep_mask], alfy[keep_mask]
    dx, dpx = dx[keep_mask], dpx[keep_mask]
    mux, muy = mux[keep_mask], muy[keep_mask]

    # Interpolation functions
    def _interp(arr):
        return interpolate.interp1d(s, arr, kind=interp_kind, fill_value="extrapolate")

    f_betx, f_bety = _interp(betx), _interp(bety)
    f_alfx, f_alfy = _interp(alfx), _interp(alfy)
    f_dx, f_dpx = _interp(dx), _interp(dpx)
    f_mux, f_muy = _interp(mux), _interp(muy)

    # Uniform grid
    s_uniform = np.linspace(0, circumference, num_interp_slice, endpoint=True)

    items = []
    name_map = {}

    for i in range(len(s_uniform)):
        si = s_uniform[i]
        if i == 0:
            tp = TwissPoint(
                s=si, s_previous=si,
                alpha_x=f_alfx(si), alpha_y=f_alfy(si),
                beta_x=f_betx(si), beta_y=f_bety(si),
                mu_x=f_mux(si), mu_y=f_muy(si), mu_z=0.0,
                dx=f_dx(si), dpx=f_dpx(si),
                alpha_x_previous=f_alfx(si), alpha_y_previous=f_alfy(si),
                beta_x_previous=f_betx(si), beta_y_previous=f_bety(si),
                mu_x_previous=f_mux(si), mu_y_previous=f_muy(si), mu_z_previous=0.0,
                dx_previous=f_dx(si), dpx_previous=f_dpx(si),
                dqx=0.0, dqy=0.0,
                longitudinal_transfer=longitudinal_transfer,
            )
        else:
            s_prev = s_uniform[i - 1]
            tp = TwissPoint(
                s=si, s_previous=s_prev,
                alpha_x=f_alfx(si), alpha_y=f_alfy(si),
                beta_x=f_betx(si), beta_y=f_bety(si),
                mu_x=f_mux(si), mu_y=f_muy(si),
                mu_z=si / circumference * muz,
                dx=f_dx(si), dpx=f_dpx(si),
                alpha_x_previous=f_alfx(s_prev), alpha_y_previous=f_alfy(s_prev),
                beta_x_previous=f_betx(s_prev), beta_y_previous=f_bety(s_prev),
                mu_x_previous=f_mux(s_prev), mu_y_previous=f_muy(s_prev),
                mu_z_previous=s_prev / circumference * muz,
                dx_previous=f_dx(s_prev), dpx_previous=f_dpx(s_prev),
                dqx=dqx * (f_mux(si) - f_mux(s_prev)) / qx,
                dqy=dqy * (f_muy(si) - f_muy(s_prev)) / qy,
                longitudinal_transfer=longitudinal_transfer,
            )
        items.append(tp)
        name_map[f"TwissInterp[{i + 1}]"] = i

    print(f"[Read MADX Twiss] {len(items)} interpolated twiss points created")

    # Insert elements
    if insert_patterns:
        insert_items, insert_names = _insert_elements(
            twiss_table, insert_patterns, circumference
        )
        items.extend(insert_items)
        for j, name in enumerate(insert_names):
            name_map[name] = len(items) - len(insert_items) + j
        print(f"[Read MADX Twiss] {len(insert_items)} thin-lens elements inserted")

    # Field errors
    if is_field_error and error_file:
        error_dict = read_madx_errors(error_file)
        error_items = []
        error_names = []
        for key, errs in error_dict.items():
            if key in name_map:
                idx = name_map[key]
                s_val = items[idx].s
                err_item = MultipoleElement(
                    s=s_val, length=0.0,
                    knl=errs["knl"], ksl=errs["ksl"],
                )
                error_items.append(err_item)
                error_names.append(f"{key}_error")
        items.extend(error_items)
        for j, name in enumerate(error_names):
            name_map[name] = len(items) - len(error_names) + j
        print(f"[Read MADX Twiss] {len(error_items)} field error multipoles added")

    return items, circumference
