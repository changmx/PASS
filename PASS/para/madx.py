"""Read MADX TFS files → PASS schema objects.

Consolidates three former modules:
    madx_element  — twiss TFS → Element list (element-by-element tracking)
    madx_twiss    — twiss TFS → TwissItem list (twiss transfer tracking)
    madx_error    — error TFS → field error dict

Element naming convention:
    f"{madx_name}_s{s:.3f}"
    e.g. "qd1_s1.200", "sd3_s15.450", "drift_s0.075"
    Merged drifts: "drift1_drift2_s0.075"
"""

import re

import numpy as np
import tfs

from PASS.para.schema.twiss import TwissItem
from PASS.para.schema.elements import DriftItem, MarkerItem, SBendItem, QuadrupoleItem, SextupoleItem, OctupoleItem, MultipoleItem, KickerItem, SolenoidItem

# Helpers


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
    Occurrence order comes only from the complete source Twiss lattice.
    The sparse error table resolves names against that source catalog.
    """
    return f"{elem_name}[{occurrence}]"


def _extract_multipole_kl(row, columns) -> tuple[list, list]:
    """Extract KNL and KSL arrays from a MADX twiss TFS row.

    MADX twiss TFS stores multipole strengths as individual K0L, K1L, K2L, ...
    and K0SL, K1SL, K2SL, ... columns (not as array-valued KNL/KSL).
    This function collects all available orders and strips trailing zeros.

    Example: K0L=0, K1L=0, K2L=0.039, K3L=0 → knl=[0, 0, 0.039]

    Args:
        row: a pandas Series (one TFS row).
        columns: the TFS column index (for membership testing).

    Returns:
        (knl, ksl) as lists of floats, with trailing zeros removed.
    """
    knl = []
    for n in range(0, 21):
        col = f"K{n}L"
        knl.append(float(row[col]) if col in columns else 0.0)

    ksl = []
    for n in range(0, 21):
        col = f"K{n}SL"
        ksl.append(float(row[col]) if col in columns else 0.0)

    # Strip trailing zeros
    while knl and knl[-1] == 0:
        knl.pop()
    while ksl and ksl[-1] == 0:
        ksl.pop()

    if not knl and not ksl:
        knl = [0.0]

    return knl, ksl


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


# Element reader (element-by-element tracking)


def merge_drift_elements(items: list, names: list[str]) -> tuple[list, list[str]]:
    """Merge consecutive DriftElements into one.

    Returns (merged_items, merged_names).
    Merged name joins the original names with '_'; S remains the exit position.
    Drifts with local SC, slicing or aperture settings retain their boundaries.
    """
    if not items:
        return [], []

    result_items = []
    result_names = []
    i = 0

    def mergeable(item):
        return (item.command == "Drift" and item.space_charge is None and item.num_slices == 1 and item.aperture_type == "off"
                and not item.aperture_value)

    while i < len(items):
        current = items[i]

        if not mergeable(current):
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
            if mergeable(items[j]):
                drift_indices.append(j)
                drift_len += items[j].length
                s_val = items[j].s
                j += 1
            else:
                break

        if len(drift_indices) == 1:
            result_items.append(current)
            result_names.append(names[i])
        else:
            merged = DriftItem(s=s_val, length=drift_len)
            merged_name = "_".join(names[k] for k in drift_indices)
            result_items.append(merged)
            result_names.append(merged_name)

        i = j

    print(f"[Read MADX Elements] Merged drifts: {len(items)} -> {len(result_items)}")
    return result_items, result_names


def merge_drift_twiss_points(items: list[TwissItem], names: list[str], keywords: list[str]) -> tuple[list[TwissItem], list[str]]:
    """Collapse consecutive MAD-X DRIFT rows into one Twiss transport point.

    The last point in each drift run is retained so its optical functions remain
    the values at the end of the run.  Its previous values and ``S previous``
    are taken from the first point, making the resulting transfer span the
    complete merged drift interval.
    """
    if not items:
        return [], []
    result_items: list[TwissItem] = []
    result_names: list[str] = []
    i = 0
    while i < len(items):
        if str(keywords[i]).casefold() != "drift":
            result_items.append(items[i])
            result_names.append(names[i])
            i += 1
            continue
        j = i + 1
        while j < len(items) and str(keywords[j]).casefold() == "drift":
            j += 1
        last = items[j - 1]
        if j - i > 1:
            first = items[i]
            last = last.model_copy(
                update={
                    "s_previous": first.s_previous,
                    "alpha_x_previous": first.alpha_x_previous,
                    "alpha_y_previous": first.alpha_y_previous,
                    "beta_x_previous": first.beta_x_previous,
                    "beta_y_previous": first.beta_y_previous,
                    "mu_x_previous": first.mu_x_previous,
                    "mu_y_previous": first.mu_y_previous,
                    "mu_z_previous": first.mu_z_previous,
                    "dx_previous": first.dx_previous,
                    "dpx_previous": first.dpx_previous,
                })
            result_names.append("_".join(names[i:j]))
        else:
            result_names.append(names[i])
        result_items.append(last)
        i = j
    print(f"[Read MADX Twiss] Merged drifts: {len(items)} -> {len(result_items)}")
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
            item = MarkerItem(s=s)
        elif et == "drift":
            item = DriftItem(s=s, length=l)
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
            item = SBendItem(
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
            item = QuadrupoleItem(
                s=s,
                length=l,
                k1l=row.get("K1L", 0.0),
                k1sl=row.get("K1SL", 0.0),
            )
        elif et == "sextupole":
            item = SextupoleItem(
                s=s,
                length=l,
                k2l=row.get("K2L", 0.0),
                k2sl=row.get("K2SL", 0.0),
            )
        elif et == "octupole":
            item = OctupoleItem(
                s=s,
                length=l,
                k3l=row.get("K3L", 0.0),
                k3sl=row.get("K3SL", 0.0),
            )
        elif et == "multipole":
            knl, ksl = _extract_multipole_kl(row, twiss_table.columns)
            item = MultipoleItem(s=s, length=l, knl=knl, ksl=ksl)
        elif et == "solenoid":
            knl, ksl = _extract_multipole_kl(row, twiss_table.columns)
            if "KSI" in twiss_table.columns:
                ksi = float(row["KSI"])
                if l == 0 and ksi != 0:
                    raise ValueError(f"Solenoid {elem_name}: nonzero KSI at zero length has no supported axial thin map")
                ks = ksi / l if l != 0 else 0.0
            elif "KS" in twiss_table.columns:
                ks = float(row["KS"])
            else:
                raise ValueError(f"Solenoid {elem_name}: export KSI in the MAD-X Twiss table (or provide KS)")
            item = SolenoidItem(s=s, length=l, ks=ks, knl=knl, ksl=ksl)
        elif et in ("hkicker", "vkicker", "kicker", "tkicker"):
            hkick = row.get("HKICK", 0.0)
            vkick = row.get("VKICK", 0.0)
            if et == "hkicker":
                hkick = hkick or row.get("K0L", 0.0)
            elif et == "vkicker":
                vkick = vkick or row.get("K0L", 0.0)
            item = KickerItem(
                s=s,
                length=l,
                hkick=hkick,
                vkick=vkick,
            )
        elif et == "monitor":
            item = DriftItem(s=s, length=l)
        else:
            print(f"[Read MADX Elements] Warning: unsupported {et} '{name}' -> drift")
            item = DriftItem(s=s, length=l)

        items.append(item)
        names.append(name)
        # Store match_key on item for error matching (survives drift merge)
        item._match_key = match_key

    # Field errors — match by name[occurrence], not by s-suffixed name
    if is_field_error:
        if not error_file:
            raise ValueError("Field-error import requires an error TFS file")
        error_dict = read_madx_errors(error_file, element_names=twiss_table["NAME"])
        key_to_idx = {}
        for idx, item in enumerate(items):
            mk = getattr(item, "_match_key", None)
            if mk is not None:
                key_to_idx[mk] = idx
        error_count = 0
        for key, errs in error_dict.items():
            if key in key_to_idx:
                idx = key_to_idx[key]
                if "is_field_error" not in type(items[idx]).model_fields:
                    raise ValueError(f"Field errors on {key!r} are not supported by {type(items[idx]).__name__}")
                items[idx].is_field_error = True
                items[idx].field_error_knl = errs["knl"]
                items[idx].field_error_ksl = errs["ksl"]
                error_count += 1
            else:
                raise ValueError(f"Field-error element {key!r} was lost while building the element sequence")
        print(f"[Read MADX Elements] {error_count} field errors attached")

    # Merge drifts
    if is_merge_drift:
        items, names = merge_drift_elements(items, names)

    # Circumference check
    length_count = sum(item.length for item in items)
    diff = length_count - circumference
    if abs(diff) < 1e-6:
        print(f"[Read MADX Elements] Circumference check passed: {length_count:.6f} m")
    else:
        print(f"[Read MADX Elements] Circumference check FAILED: "
              f"theory={circumference}, actual={length_count}, diff={diff:.6e}")

    return items, names, circumference


# Twiss reader (twiss transfer tracking)


def read_madx_twiss(
    twiss_file: str,
    error_file: str = "",
    muz: float = 0.0,
    dqx: float | str = "from_file",
    dqy: float | str = "from_file",
    is_field_error: bool = False,
    insert_patterns: list[str] | None = None,
    longitudinal_transfer: str = "off",
    is_merge_drift: bool = False,
) -> tuple[list, list[str], float]:
    """Read a MADX twiss TFS file → (twiss_items, item_names, circumference).

    Each row becomes a TwissItem with current + previous optical functions.
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
        is_merge_drift: merge consecutive Drift elements if present in the result.
        insert_patterns: regex patterns to match element names for thin-lens insertion.
        longitudinal_transfer: "off" / "drift" / "matrix".

    Returns:
        (items, names, circumference) where items is a list of TwissItem
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
    keywords = []
    name_count = {}

    for i in range(num_elem):
        elem_name = twiss_table.iloc[i]["NAME"]
        name = "twiss_" + _make_name(elem_name, s[i])
        name_count[elem_name] = name_count.get(elem_name, 0) + 1
        match_key = _make_match_key(elem_name, name_count[elem_name])

        if i == 0:
            tp = TwissItem(
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
            tp = TwissItem(
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
        keywords.append(str(twiss_table.iloc[i].get("KEYWORD", "")))
        tp._match_key = match_key

    print(f"[Read MADX Twiss] {len(items)} twiss points created")

    error_dict = {}
    if is_field_error:
        if not error_file:
            raise ValueError("Field-error import requires an error TFS file")
        error_dict = read_madx_errors(error_file, element_names=twiss_table["NAME"])
        # A localized error must retain its transport endpoint through merging.
        keywords = ["field_error" if item._match_key in error_dict else keyword for item, keyword in zip(items, keywords)]

    if is_merge_drift:
        items, names = merge_drift_twiss_points(items, names, keywords)

    # --- Insert thin-lens elements ---
    if insert_patterns:
        insert_items, insert_names = _insert_elements(twiss_table, insert_patterns)
        items.extend(insert_items)
        names.extend(insert_names)
        print(f"[Read MADX Twiss] {len(insert_items)} thin-lens elements inserted")

    # --- Attach field errors ---
    if is_field_error:
        error_items = []
        error_names = []
        source_positions, occurrences = {}, {}
        for _, row in twiss_table.iterrows():
            raw_name = str(row["NAME"])
            occurrences[raw_name] = occurrences.get(raw_name, 0) + 1
            source_positions[_make_match_key(raw_name, occurrences[raw_name])] = float(row["S"])
        for key, errs in error_dict.items():
            if key in source_positions:
                s_val = source_positions[key]
                err_item = MultipoleItem(
                    s=s_val,
                    length=0.0,
                    knl=errs["knl"],
                    ksl=errs["ksl"],
                )
                error_items.append(err_item)
                error_names.append(f"{key}_error")
            else:
                raise ValueError(f"Field-error element {key!r} was lost while building the Twiss sequence")
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
    """Create thin-lens elements for names matching *insert_patterns*.

    TwissItem names from read_madx_twiss are prefixed with 'twiss_', so
    there is no name collision with inserted elements.
    """
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
            item = QuadrupoleItem(
                s=s,
                length=0.0,
                k1l=twiss_table.iloc[i].get("K1L", 0.0),
                k1sl=twiss_table.iloc[i].get("K1SL", 0.0),
            )
        elif et == "sextupole":
            item = SextupoleItem(
                s=s,
                length=0.0,
                k2l=twiss_table.iloc[i].get("K2L", 0.0),
                k2sl=twiss_table.iloc[i].get("K2SL", 0.0),
            )
        elif et == "octupole":
            item = OctupoleItem(
                s=s,
                length=0.0,
                k3l=twiss_table.iloc[i].get("K3L", 0.0),
                k3sl=twiss_table.iloc[i].get("K3SL", 0.0),
            )
        elif et == "multipole":
            knl, ksl = _extract_multipole_kl(twiss_table.iloc[i], twiss_table.columns)
            item = MultipoleItem(s=s, length=0.0, knl=knl, ksl=ksl)
        elif et in ("hkicker", "vkicker", "kicker", "tkicker"):
            item = KickerItem(
                s=s,
                length=0.0,
                hkick=twiss_table.iloc[i].get("HKICK", 0.0),
                vkick=twiss_table.iloc[i].get("VKICK", 0.0),
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
    muz: float = 0.0,
    dqx: float | str = "from_file",
    dqy: float | str = "from_file",
    is_field_error: bool = False,
    insert_patterns: list[str] | None = None,
    longitudinal_transfer: str = "off",
    interp_kind: str = "phase_hermite",
) -> tuple[list, list[str], float]:
    """Resample a full ring using phase-constrained quintic Hermite optics.

    ``num_interp_slice`` counts base points including 0 and C: N segments
    require N+1 points. Original rows are replaced; kicks, field errors and
    optical discontinuities add split points. Source phases and their full
    tune are preserved. No extrapolation is allowed. Only
    ``interp_kind='phase_hermite'`` is supported. DQx/DQy default to TFS headers.
    """
    from PASS.para.twiss_interpolation import resample_madx_twiss

    return resample_madx_twiss(twiss_file, num_interp_slice, error_file, muz, dqx, dqy, is_field_error, insert_patterns, longitudinal_transfer,
                               interp_kind)


# Error reader


def _error_instance_catalog(element_names):
    """Build instance identities once from the unmerged source lattice."""
    counts = {}
    exact = {}
    instances = {}
    bases = {}
    for raw in element_names:
        raw = str(raw)
        counts[raw] = counts.get(raw, 0) + 1
        key = _make_match_key(raw, counts[raw])
        exact.setdefault(raw.casefold(), []).append(key)
        match = re.fullmatch(r"(.+?)(?:\[(\d+)\]|:(\d+))", raw)
        if match:
            base, occurrence = match[1], int(match[2] or match[3])
        else:
            base, occurrence = raw, counts[raw]
        instances.setdefault((base.casefold(), occurrence), []).append(key)
        bases.setdefault(base.casefold(), []).append(key)
    return exact, instances, bases


def _match_error_instance(name, catalog):
    """Sparse error-table row order cannot identify repeated lattice instances."""
    exact, instances, bases = catalog
    candidates = exact.get(name.casefold(), [])
    if not candidates:
        match = re.fullmatch(r"(.+?)(?:\[(\d+)\]|:(\d+))", name)
        if match:
            candidates = instances.get((match[1].casefold(), int(match[2] or match[3])), [])
        else:
            candidates = bases.get(name.casefold(), [])
    if len(candidates) != 1:
        reason = "ambiguous" if candidates else "not found"
        raise ValueError(f"Field-error element {name!r} is {reason} in the source Twiss table; "
                         "use unique MAD-X instance names or an explicit NAME occurrence such as Q[2]")
    return candidates[0]


def read_madx_errors(error_file_path: str, *, element_names=None) -> dict[str, dict]:
    """Read absolute integrated K<n>L/K<n>SL errors, preserving every finite value.

    Missing orders are zero. Without source names only unique error-table names
    can be checked; importers must pass the original, unmerged Twiss NAME column.
    Non-field columns (alignment, aperture and monitor errors) are not imported.
    """
    table = tfs.read(error_file_path)
    if "NAME" not in table.columns:
        raise ValueError("Field-error TFS must contain a NAME column")
    columns = []
    for column in table.columns:
        match = re.fullmatch(r"K(\d+)(S?)L", str(column))
        if match:
            columns.append((column, int(match[1]), bool(match[2])))
    if not columns:
        raise ValueError("Field-error TFS must contain K<n>L or K<n>SL columns")
    names = list(table["NAME"] if element_names is None else element_names)
    catalog = _error_instance_catalog(names)
    result = {}
    seen = set()
    for row_index, (_, row) in enumerate(table.iterrows(), start=1):
        name = str(row["NAME"])
        n = max(order for _, order, _ in columns) + 1
        knl, ksl = np.zeros(n), np.zeros(n)
        for column, order, skew in columns:
            value = float(row[column])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite field error at row {row_index}, {name!r}, column {column}")
            (ksl if skew else knl)[order] = value
        nonzero = np.flatnonzero((knl != 0) | (ksl != 0))
        if not len(nonzero):
            continue
        key = _match_error_instance(name, catalog)
        if key in seen:
            raise ValueError(f"Duplicate field-error records for {key!r}; refusing to overwrite or add them")
        seen.add(key)
        n = int(nonzero[-1]) + 1
        result[key] = {"knl": knl[:n].tolist(), "ksl": ksl[:n].tolist()}
    print(f"[Read MADX Errors] {len(table)} rows, {len(result)} nonzero absolute field errors")
    return result
