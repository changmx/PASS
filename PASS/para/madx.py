"""Read MADX TFS files → PASS schema objects.

Consolidates three former modules:
    madx_element  — twiss TFS → Element list (element-by-element tracking)
    madx_twiss    — twiss TFS → TwissPoint list (twiss transfer tracking)
    madx_error    — error TFS → field error dict

Element naming convention:
    f"{madx_name}_s{s:.3f}"
    e.g. "qd1_s1.200", "sd3_s15.450", "drift_s0.075"
    Merged drifts: "drift1_drift2_s0.075"
"""

import re
import numpy as np
import tfs
from scipy import interpolate

from PASS.para.schema.twiss import TwissPoint
from PASS.para.schema.elements import (
    DriftElement,
    MarkerElement,
    SBendElement,
    QuadrupoleElement,
    SextupoleElement,
    OctupoleElement,
    MultipoleElement,
    KickerElement,
)

# ============================================================
# Helpers
# ============================================================


def _make_name(elem_name: str, s: float) -> str:
    """Build a unique name: f"{madx_name}_s{s:.3f}".

    Uses the raw MADX element name (not occurrence-suffixed) so the
    original lattice label is preserved.  The s-position suffix makes
    every name unique even when the same MADX name appears multiple times.
    """
    return f"{elem_name}_s{s:.3f}"


def _make_match_key(elem_name: str, occurrence: int) -> str:
    """Build a matching key: f"{madx_name}[{occurrence}]".

    Used for error-to-element matching, which must be independent of
    the S column (error TFS S values may differ from twiss TFS).
    Both the element reader and error reader track occurrence order
    so the same name always produces the same key.
    """
    return f"{elem_name}[{occurrence}]"


def _read_tfs_headers(twiss_file: str) -> dict:
    """Read MADX twiss TFS headers + first-row twiss parameters."""
    df = tfs.read(twiss_file)
    headers = df.headers
    row0 = df.iloc[0]
    return {
        "df": df,
        "circumference": headers["LENGTH"],
        "gamma": headers["GAMMA"],
        "gamma_tr": headers["GAMMATR"],
        "q1": headers["Q1"],
        "q2": headers["Q2"],
        "dq1": headers["DQ1"],
        "dq2": headers["DQ2"],
        "betx": row0["BETX"],
        "alfx": row0["ALFX"],
        "bety": row0["BETY"],
        "alfy": row0["ALFY"],
        "dx": row0["DX"],
        "dpx": row0["DPX"],
        "energy": headers["ENERGY"],
        "mass": headers["MASS"],
    }


# ============================================================
# Element reader (element-by-element tracking)
# ============================================================


def merge_drift_elements(items: list, names: list[str]) -> tuple[list, list[str]]:
    """Merge consecutive DriftElements into one.

    Returns (merged_items, merged_names).
    Merged name joins the original names with '_'.
    """
    if not items:
        return [], []

    result_items = []
    result_names = []
    i = 0

    while i < len(items):
        current = items[i]

        if current.command != "Drift":
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

    print(f"[Read MADX Elements] Merged drifts: {len(items)} -> {len(result_items)}")
    return result_items, result_names


def read_madx_elements(
    twiss_file: str,
    error_file: str = "",
    is_merge_drift: bool = False,
    is_field_error: bool = False,
) -> tuple[list, list[str], float]:
    """Read a MADX twiss TFS file → (element_items, element_names, circumference).

    Each MADX element is converted to the corresponding PASS Element schema
    object with its physical parameters (length, strength, edge angles, etc.).

    Naming: f"{madx_name}_s{s:.3f}" (e.g. "qd1_s1.200").

    SBend K0L is automatically patched from the ANGLE column when K0L is
    zero or absent (MADX twiss TFS stores the bend angle in ANGLE, not K0L).

    Args:
        twiss_file: path to MADX twiss TFS file.
        error_file: path to MADX error TFS file.
        is_merge_drift: merge consecutive drift elements.
        is_field_error: attach field errors to matching elements.

    Returns:
        (items, names, circumference) where items is a list of Element
        schema objects and names is a list of corresponding string names.
    """
    twiss_table = tfs.read(twiss_file)
    num_elem = twiss_table.shape[0]
    circumference = twiss_table.headers["LENGTH"]
    print(f"[Read MADX Elements] {num_elem} elements, C={circumference}")

    items = []
    names = []
    name_count = {}

    for i in range(num_elem):
        row = twiss_table.iloc[i]
        elem_name = row["NAME"]
        elem_type = row["KEYWORD"]
        s = row["S"]
        l = row["L"]

        name = _make_name(elem_name, s)
        name_count[elem_name] = name_count.get(elem_name, 0) + 1
        match_key = _make_match_key(elem_name, name_count[elem_name])
        et = elem_type.lower()

        if et == "marker":
            item = MarkerElement(s=s)
        elif et == "drift":
            item = DriftElement(s=s, length=l)
        elif et in ("sbend", "rbend"):
            fint = row.get("FINT", 0.0)
            fintx = row.get("FINTX", 0.0)
            if fintx <= 0:
                fintx = fint
            # Patch K0L from ANGLE column (MADX twiss TFS stores angle, not K0L)
            k0l = row.get("K0L", 0.0)
            angle = row.get("ANGLE", 0.0)
            if abs(k0l) < 1e-15 and abs(angle) > 1e-15:
                k0l = angle
            item = SBendElement(
                s=s,
                length=l,
                k0l=k0l,
                e1=row.get("E1", 0.0),
                e2=row.get("E2", 0.0),
                hgap=row.get("HGAP", 0.0),
                fint=fint,
                fintx=fintx,
            )
        elif et == "quadrupole":
            item = QuadrupoleElement(
                s=s,
                length=l,
                k1l=row.get("K1L", 0.0),
                k1sl=row.get("K1SL", 0.0),
            )
        elif et == "sextupole":
            item = SextupoleElement(
                s=s,
                length=l,
                k2l=row.get("K2L", 0.0),
                k2sl=row.get("K2SL", 0.0),
            )
        elif et == "octupole":
            item = OctupoleElement(
                s=s,
                length=l,
                k3l=row.get("K3L", 0.0),
                k3sl=row.get("K3SL", 0.0),
            )
        elif et == "multipole":
            item = MultipoleElement(s=s, length=l, knl=[], ksl=[])
        elif et in ("hkicker", "vkicker", "kicker", "tkicker"):
            hkick = row.get("HKICK", 0.0)
            vkick = row.get("VKICK", 0.0)
            if et == "hkicker":
                hkick = hkick or row.get("K0L", 0.0)
            elif et == "vkicker":
                vkick = vkick or row.get("K0L", 0.0)
            item = KickerElement(
                s=s,
                length=l,
                hkick=hkick,
                vkick=vkick,
            )
        elif et == "monitor":
            item = DriftElement(s=s, length=l)
        else:
            print(f"[Read MADX Elements] Warning: unsupported {et} '{name}' -> drift")
            item = DriftElement(s=s, length=l)

        items.append(item)
        names.append(name)
        # Store match_key on item for error matching (survives drift merge)
        item._match_key = match_key

    # Merge drifts
    if is_merge_drift:
        items, names = merge_drift_elements(items, names)

    # Field errors — match by name[occurrence], not by s-suffixed name
    if is_field_error and error_file:
        error_dict = read_madx_errors(error_file)
        key_to_idx = {}
        for idx, item in enumerate(items):
            mk = getattr(item, "_match_key", None)
            if mk is not None:
                key_to_idx[mk] = idx
        error_count = 0
        for key, errs in error_dict.items():
            if key in key_to_idx:
                idx = key_to_idx[key]
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

    return items, names, circumference


# ============================================================
# Twiss reader (twiss transfer tracking)
# ============================================================


def read_madx_twiss(
    twiss_file: str,
    error_file: str = "",
    muz: float = 0.0,
    dqx: float | str = "from_file",
    dqy: float | str = "from_file",
    is_field_error: bool = False,
    insert_patterns: list[str] | None = None,
    longitudinal_transfer: str = "off",
) -> tuple[list, list[str], float]:
    """Read a MADX twiss TFS file → (twiss_items, item_names, circumference).

    Each row becomes a TwissPoint with current + previous optical functions.
    The first point has previous = current.

    Optionally inserts thin-lens elements (quad/sext/oct/kicker/multipole)
    matched by *insert_patterns* (regex) alongside the twiss points.

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
        (items, names, circumference) where items is a list of TwissPoint
        and optionally Element objects, and names is the corresponding
        list of string names.
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
    names = []
    name_count = {}

    for i in range(num_elem):
        elem_name = twiss_table.iloc[i]["NAME"]
        name = _make_name(elem_name, s[i])
        name_count[elem_name] = name_count.get(elem_name, 0) + 1
        match_key = _make_match_key(elem_name, name_count[elem_name])

        if i == 0:
            tp = TwissPoint(
                s=s[i],
                s_previous=s[i],
                alpha_x=alfx[i],
                alpha_y=alfy[i],
                beta_x=betx[i],
                beta_y=bety[i],
                mu_x=mux[i],
                mu_y=muy[i],
                mu_z=0.0,
                dx=dx[i],
                dpx=dpx[i],
                alpha_x_previous=alfx[i],
                alpha_y_previous=alfy[i],
                beta_x_previous=betx[i],
                beta_y_previous=bety[i],
                mu_x_previous=mux[i],
                mu_y_previous=muy[i],
                mu_z_previous=0.0,
                dx_previous=dx[i],
                dpx_previous=dpx[i],
                dqx=0.0,
                dqy=0.0,
                longitudinal_transfer=longitudinal_transfer,
            )
        else:
            mu_z_i = s[i] / circumference * muz
            mu_z_prev = s[i - 1] / circumference * muz
            tp = TwissPoint(
                s=s[i],
                s_previous=s[i - 1],
                alpha_x=alfx[i],
                alpha_y=alfy[i],
                beta_x=betx[i],
                beta_y=bety[i],
                mu_x=mux[i],
                mu_y=muy[i],
                mu_z=mu_z_i,
                dx=dx[i],
                dpx=dpx[i],
                alpha_x_previous=alfx[i - 1],
                alpha_y_previous=alfy[i - 1],
                beta_x_previous=betx[i - 1],
                beta_y_previous=bety[i - 1],
                mu_x_previous=mux[i - 1],
                mu_y_previous=muy[i - 1],
                mu_z_previous=mu_z_prev,
                dx_previous=dx[i - 1],
                dpx_previous=dpx[i - 1],
                dqx=dqx * (mux[i] - mux[i - 1]) / qx,
                dqy=dqy * (muy[i] - muy[i - 1]) / qy,
                longitudinal_transfer=longitudinal_transfer,
            )
        items.append(tp)
        names.append(name)
        tp._match_key = match_key

    print(f"[Read MADX Twiss] {len(items)} twiss points created")

    # --- Insert thin-lens elements ---
    if insert_patterns:
        insert_items, insert_names = _insert_elements(twiss_table, insert_patterns)
        items.extend(insert_items)
        names.extend(insert_names)
        print(f"[Read MADX Twiss] {len(insert_items)} thin-lens elements inserted")

    # --- Attach field errors ---
    if is_field_error and error_file:
        error_dict = read_madx_errors(error_file)
        error_items = []
        error_names = []
        key_to_idx = {}
        for idx, item in enumerate(items):
            mk = getattr(item, "_match_key", None)
            if mk is not None:
                key_to_idx[mk] = idx
        for key, errs in error_dict.items():
            if key in key_to_idx:
                idx = key_to_idx[key]
                s_val = items[idx].s if hasattr(items[idx], "s") else \
                    items[idx].model_dump(by_alias=True)["S (m)"]
                err_item = MultipoleElement(
                    s=s_val,
                    length=0.0,
                    knl=errs["knl"],
                    ksl=errs["ksl"],
                )
                error_items.append(err_item)
                error_names.append(f"{key}_error")
            else:
                print(f"[Read MADX Twiss] Warning: error element '{key}' not found in twiss")
        items.extend(error_items)
        names.extend(error_names)
        print(f"[Read MADX Twiss] {len(error_items)} field error multipoles added")

    # --- Circumference check ---
    length_count = 0.0
    for item in items:
        d = item.model_dump(by_alias=True)
        if "Length (m)" in d and d["Length (m)"] > 0:
            length_count += d["Length (m)"]
        elif "S previous (m)" in d:
            length_count += d["S (m)"] - d["S previous (m)"]

    diff = length_count - circumference
    if abs(diff) < 1e-6:
        print(f"[Read MADX Twiss] Circumference check passed: {length_count:.6f} m")
    else:
        print(f"[Read MADX Twiss] Circumference check FAILED: "
              f"theory={circumference}, actual={length_count}, diff={diff:.6e}")

    return items, names, circumference


def _insert_elements(twiss_table, insert_patterns: list[str]) -> tuple[list, list[str]]:
    """Create thin-lens elements for names matching *insert_patterns*."""
    combined = re.compile("|".join(f"({p})" for p in insert_patterns))
    items = []
    names = []

    for i in range(len(twiss_table)):
        elem_name = twiss_table.iloc[i]["NAME"]
        elem_type = twiss_table.iloc[i]["KEYWORD"]
        s = twiss_table.iloc[i]["S"]
        name = _make_name(elem_name, s)

        if not combined.search(name):
            continue

        et = elem_type.lower()

        if et == "quadrupole":
            item = QuadrupoleElement(
                s=s,
                length=0.0,
                k1l=twiss_table.iloc[i]["K1L"],
                k1sl=twiss_table.iloc[i]["K1SL"],
            )
        elif et == "sextupole":
            item = SextupoleElement(
                s=s,
                length=0.0,
                k2l=twiss_table.iloc[i]["K2L"],
                k2sl=twiss_table.iloc[i]["K2SL"],
            )
        elif et == "octupole":
            item = OctupoleElement(
                s=s,
                length=0.0,
                k3l=twiss_table.iloc[i]["K3L"],
                k3sl=twiss_table.iloc[i]["K3SL"],
            )
        elif et == "multipole":
            item = MultipoleElement(s=s, length=0.0, knl=[], ksl=[])
        elif et in ("hkicker", "vkicker", "kicker", "tkicker"):
            item = KickerElement(
                s=s,
                length=0.0,
                hkick=twiss_table.iloc[i]["HKICK"],
                vkick=twiss_table.iloc[i]["VKICK"],
            )
        else:
            print(f"[Read MADX Twiss] Warning: cannot insert {et} '{name}', skipping")
            continue

        items.append(item)
        names.append(name)

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
) -> tuple[list, list[str], float]:
    """Read MADX twiss and interpolate onto a uniform s-grid.

    Produces num_interp_slice TwissPoints evenly spaced along the ring.
    Insert elements (if any) are placed at their original s positions.
    """
    twiss_table = tfs.read(twiss_file)
    headers = twiss_table.headers
    circumference = headers["LENGTH"]
    qx = headers["Q1"]
    qy = headers["Q2"]

    if dqx == "from_file":
        dqx = headers["DQ1"]
    if dqy == "from_file":
        dqy = headers["DQ2"]

    print(f"[Read MADX Twiss] Interpolated: {num_interp_slice} slices, "
          f"C={circumference}, Qx={qx}, Qy={qy}")

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

    def _interp(arr):
        return interpolate.interp1d(s, arr, kind=interp_kind, fill_value="extrapolate")

    f_betx, f_bety = _interp(betx), _interp(bety)
    f_alfx, f_alfy = _interp(alfx), _interp(alfy)
    f_dx, f_dpx = _interp(dx), _interp(dpx)
    f_mux, f_muy = _interp(mux), _interp(muy)

    s_uniform = np.linspace(0, circumference, num_interp_slice, endpoint=True)

    items = []
    names = []

    for i in range(len(s_uniform)):
        si = s_uniform[i]
        name = f"twiss_interp_s{si:.3f}"

        if i == 0:
            tp = TwissPoint(
                s=si,
                s_previous=si,
                alpha_x=f_alfx(si),
                alpha_y=f_alfy(si),
                beta_x=f_betx(si),
                beta_y=f_bety(si),
                mu_x=f_mux(si),
                mu_y=f_muy(si),
                mu_z=0.0,
                dx=f_dx(si),
                dpx=f_dpx(si),
                alpha_x_previous=f_alfx(si),
                alpha_y_previous=f_alfy(si),
                beta_x_previous=f_betx(si),
                beta_y_previous=f_bety(si),
                mu_x_previous=f_mux(si),
                mu_y_previous=f_muy(si),
                mu_z_previous=0.0,
                dx_previous=f_dx(si),
                dpx_previous=f_dpx(si),
                dqx=0.0,
                dqy=0.0,
                longitudinal_transfer=longitudinal_transfer,
            )
        else:
            s_prev = s_uniform[i - 1]
            tp = TwissPoint(
                s=si,
                s_previous=s_prev,
                alpha_x=f_alfx(si),
                alpha_y=f_alfy(si),
                beta_x=f_betx(si),
                beta_y=f_bety(si),
                mu_x=f_mux(si),
                mu_y=f_muy(si),
                mu_z=si / circumference * muz,
                dx=f_dx(si),
                dpx=f_dpx(si),
                alpha_x_previous=f_alfx(s_prev),
                alpha_y_previous=f_alfy(s_prev),
                beta_x_previous=f_betx(s_prev),
                beta_y_previous=f_bety(s_prev),
                mu_x_previous=f_mux(s_prev),
                mu_y_previous=f_muy(s_prev),
                mu_z_previous=s_prev / circumference * muz,
                dx_previous=f_dx(s_prev),
                dpx_previous=f_dpx(s_prev),
                dqx=dqx * (f_mux(si) - f_mux(s_prev)) / qx,
                dqy=dqy * (f_muy(si) - f_muy(s_prev)) / qy,
                longitudinal_transfer=longitudinal_transfer,
            )
        items.append(tp)
        names.append(name)

    print(f"[Read MADX Twiss] {len(items)} interpolated twiss points created")

    # Insert elements
    if insert_patterns:
        insert_items, insert_names = _insert_elements(twiss_table, insert_patterns)
        items.extend(insert_items)
        names.extend(insert_names)
        print(f"[Read MADX Twiss] {len(insert_items)} thin-lens elements inserted")

    # Field errors
    if is_field_error and error_file:
        error_dict = read_madx_errors(error_file)
        error_items = []
        error_names = []
        key_to_idx = {}
        for idx, item in enumerate(items):
            mk = getattr(item, "_match_key", None)
            if mk is not None:
                key_to_idx[mk] = idx
        for key, errs in error_dict.items():
            if key in key_to_idx:
                idx = key_to_idx[key]
                s_val = items[idx].s
                err_item = MultipoleElement(
                    s=s_val,
                    length=0.0,
                    knl=errs["knl"],
                    ksl=errs["ksl"],
                )
                error_items.append(err_item)
                error_names.append(f"{key}_error")
        items.extend(error_items)
        names.extend(error_names)
        print(f"[Read MADX Twiss] {len(error_items)} field error multipoles added")

    return items, names, circumference


# ============================================================
# Error reader
# ============================================================


def read_madx_errors(error_file_path: str) -> dict[str, dict]:
    """Read field errors from a MADX error TFS file.

    The error file must contain columns K0L, K1L, ..., K20L and
    K0SL, K1SL, ..., K20SL (standard MADX EFCOMP output).

    Args:
        error_file_path: path to the MADX error TFS file.

    Returns:
        {match_key: {"knl": [k0l, k1l, ...], "ksl": [k0sl, k1sl, ...]}}
        Only elements with non-zero errors are included.
        The key uses f"{madx_name}[{occurrence}]" format, matching the
        element reader's _make_match_key() so errors can be matched
        regardless of S column values.
    """
    error_table = tfs.read(error_file_path)
    num_elem = error_table.shape[0]

    print(f"[Read MADX Errors] {num_elem} elements in error file, "
          f"first='{error_table.iloc[0]['NAME']}', last='{error_table.iloc[-1]['NAME']}'")

    error_dict = {}
    name_count = {}

    for i in range(num_elem):
        elem_name = error_table.iloc[i]["NAME"]
        name_count[elem_name] = name_count.get(elem_name, 0) + 1
        match_key = _make_match_key(elem_name, name_count[elem_name])

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

            error_dict[match_key] = {"knl": knl, "ksl": ksl}

    print(f"[Read MADX Errors] {len(error_dict)} elements with non-zero errors")
    return error_dict
