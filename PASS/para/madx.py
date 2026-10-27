"""Read MADX TFS files or computed tables → PASS schema objects.

Consolidates three former modules:
    madx_element  — twiss TFS → Element list (element-by-element tracking)
    madx_twiss    — twiss TFS → TwissItem list (twiss transfer tracking)
    madx_error    — error TFS → magnetic field/alignment error dict

Element naming convention:
    f"{madx_name}_s{s:.3f}"
    e.g. "qd1_s1.200", "sd3_s15.450", "drift_s0.075"
    Merged drifts: "drift1_drift2_s0.075"
"""

import bisect
import math
import re
import warnings

import numpy as np
import tfs

from PASS.para.bpm import _create_bpm_items, _is_bpm, _select_bpm_rows, _split_bpm_twiss_table, _unique_names
from PASS.para.schema.twiss import TwissItem
from PASS.para.schema.elements import DriftItem, MarkerItem, SBendItem, QuadrupoleItem, SextupoleItem, OctupoleItem, MultipoleItem, KickerItem, SolenoidItem
from PASS.utils.command_order import command_position_key
from PASS.utils.constants import const

# Helpers


def read_madx_table(source):
    """Read a TFS path or copy an in-memory table without mutating cached optics."""
    if isinstance(source, tfs.TfsDataFrame):
        table = source.copy(deep=True)
        table.headers = dict(source.headers)
        return table
    return tfs.read(source)


def _geometry_tolerance(atol: float) -> float:
    """Use an absolute metre tolerance without a circumference-relative allowance."""
    atol = float(atol)
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("Geometry tolerance must be finite and nonnegative")
    return atol


def _finite_geometry_value(value, label: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: expected a finite value in metres, actual={value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite value in metres, actual={value!r}")
    return value


def _require_equal(actual: float, expected: float, label: str, atol: float) -> None:
    delta = actual - expected
    if abs(delta) > atol:
        raise ValueError(f"{label}: expected={expected:.17g} m, actual={actual:.17g} m, delta={delta:.6e} m, tolerance={atol:.6e} m")


def _require_at_least(actual: float, lower: float, label: str, atol: float) -> None:
    if actual < lower - atol:
        raise ValueError(f"{label}: expected>={lower:.17g} m, actual={actual:.17g} m, delta={actual - lower:.6e} m, tolerance={atol:.6e} m")


def _require_ring_bounds(position: float, circumference: float, label: str, atol: float) -> None:
    _require_at_least(position, 0.0, label, atol)
    if position > circumference + atol:
        raise ValueError(f"{label}: expected<={circumference:.17g} m, actual={position:.17g} m, "
                         f"delta={position - circumference:.6e} m, tolerance={atol:.6e} m")


def _validated_circumference(value, label: str) -> float:
    circumference = _finite_geometry_value(value, label)
    if circumference <= 0:
        raise ValueError(f"{label}: expected>0 m, actual={circumference:.17g} m")
    return circumference


def _drifts_are_contiguous(current_end: float, next_start: float, atol: float = 1e-6) -> bool:
    """Keep gaps, overlaps and nonfinite coordinates out of merged drift bodies."""
    atol = _geometry_tolerance(atol)
    return math.isfinite(current_end) and math.isfinite(next_start) and abs(current_end - next_start) <= atol


def _validate_madx_geometry(table, *, element_mode: bool, atol: float = 1e-6) -> None:
    """Validate full element geometry or sampled Twiss coverage of one ring.

    MAD-X ``S`` is an element exit. Element tracking therefore needs every body
    represented, while sampled Twiss transport only needs ordered ring coverage.
    """
    atol = _geometry_tolerance(atol)
    if "LENGTH" not in table.headers:
        raise ValueError("MAD-X geometry: missing LENGTH header for the ring circumference")
    circumference = _validated_circumference(table.headers["LENGTH"], "MAD-X geometry LENGTH header")
    for column in ("S", "L") if element_mode else ("S", ):
        if column not in table.columns:
            raise ValueError(f"MAD-X geometry: missing {column} column")
    if table.empty:
        raise ValueError("MAD-X geometry: the source table has no rows")

    previous_s = 0.0
    length_sum, compensation = 0.0, 0.0
    for index, (_, row) in enumerate(table.iterrows()):
        label = f"MAD-X geometry row {index + 1} ({str(row.get('NAME', '<unnamed>'))!r})"
        s = _finite_geometry_value(row["S"], f"{label} S")
        _require_ring_bounds(s, circumference, f"{label} S", 0.0)
        _require_at_least(s, previous_s, f"{label} S must be nondecreasing", 0.0)
        if index == 0 and not element_mode:
            _require_equal(s, 0.0, f"{label} first Twiss S must start the ring", atol)
        if "L" in table.columns:
            length = _finite_geometry_value(row["L"], f"{label} L")
            _require_at_least(length, 0.0, f"{label} L", 0.0)
            _require_ring_bounds(s - length, circumference, f"{label} body entrance S-L", atol)
            if element_mode:
                # Compensated summation avoids accumulating drift-length roundoff.
                corrected_length = length - compensation
                next_sum = length_sum + corrected_length
                compensation = (next_sum - length_sum) - corrected_length
                length_sum = next_sum
                _require_equal(s, length_sum, f"{label} exit S must match cumulative L", atol)
        previous_s = s

    _require_equal(previous_s, circumference, f"{label} final S must match LENGTH", atol)
    if element_mode:
        _require_equal(length_sum, circumference, "MAD-X geometry total L must match LENGTH", atol)


def _validate_imported_geometry(items: list, names: list[str], circumference: float, *, element_mode: bool, atol: float = 1e-6) -> None:
    """Check converted bodies or a Twiss interval chain plus its point commands."""
    atol = _geometry_tolerance(atol)
    circumference = _validated_circumference(circumference, "Imported geometry circumference")
    if len(items) != len(names):
        raise ValueError(f"Imported geometry: item/name counts differ ({len(items)} items, {len(names)} names)")
    if not items:
        raise ValueError("Imported geometry: no transport items cover the ring")

    previous_s = 0.0
    length_sum, compensation = 0.0, 0.0
    transport_ends, point_nodes = [0.0], []
    n_transports = 0
    last_transport_label = "Imported geometry"
    for index, (item, name) in enumerate(zip(items, names)):
        label = f"Imported geometry item {index + 1} ({str(name)!r}, {item.command})"
        s = _finite_geometry_value(item.s, f"{label} S")
        _require_ring_bounds(s, circumference, f"{label} S", 0.0)
        length = _finite_geometry_value(getattr(item, "length", 0.0), f"{label} length")
        _require_at_least(length, 0.0, f"{label} length", 0.0)
        if element_mode:
            if item.command == "Twiss":
                raise ValueError(f"{label}: Twiss transport cannot be used in element geometry")
            _require_at_least(s, previous_s, f"{label} S must be nondecreasing", 0.0)
            corrected_length = length - compensation
            next_sum = length_sum + corrected_length
            compensation = (next_sum - length_sum) - corrected_length
            length_sum = next_sum
            _require_equal(s, length_sum, f"{label} exit S must match cumulative length", atol)
            previous_s = s
            last_transport_label = label
        elif item.command == "Twiss":
            start = _finite_geometry_value(item.s_previous, f"{label} S previous")
            _require_ring_bounds(start, circumference, f"{label} S previous", 0.0)
            _require_at_least(s, start, f"{label} transport interval must be forward", 0.0)
            _require_at_least(s, previous_s, f"{label} S must be nondecreasing", 0.0)
            _require_equal(start, previous_s, f"{label} transport entrance must match previous exit", atol)
            previous_s = s
            last_transport_label = label
            transport_ends.append(s)
            n_transports += 1
        else:
            _require_equal(length, 0.0, f"{label} point command must have zero length in Twiss mode", 0.0)
            point_nodes.append((s, label))

    if not element_mode and n_transports == 0:
        raise ValueError("Imported geometry: no Twiss transport intervals cover the ring")
    _require_equal(previous_s, circumference, f"{last_transport_label} final transport exit must match circumference", atol)
    if element_mode:
        _require_equal(length_sum, circumference, "Imported geometry total length must match circumference", atol)
    else:
        # Diagnostics and thin kicks must execute at a split transport boundary.
        transport_ends.sort()
        endpoint_keys = {command_position_key({"s": endpoint}) for endpoint in transport_ends}
        for s, label in point_nodes:
            if command_position_key({"s": s}) in endpoint_keys:
                continue
            index = bisect.bisect_left(transport_ends, s)
            candidates = transport_ends[max(0, index - 1):index + 1]
            closest = min(candidates, key=lambda endpoint: abs(endpoint - s))
            raise ValueError(f"{label} point S must share a runtime position key with a Twiss transport endpoint: "
                             f"expected={closest:.17g} m, actual={s:.17g} m, delta={s - closest:.6e} m, "
                             f"runtime position tolerance={const.eps:.6e} m")


def _get_madx_reference_beta(headers: dict) -> float:
    """Resolve the reference speed for native TWISS PT derivatives."""
    controlled_ptc = (headers.get("PASS_SOURCE") == "MADX_SEQUENCE" and headers.get("PASS_DERIVATIVE") == "PT"
                      and headers.get("PASS_PTC_TIME") in (True, 1))
    if str(headers.get("NAME", "")).upper() == "PTC_TWISS" and not controlled_ptc:
        raise ValueError("PTC_TWISS derivatives depend on the PTC TIME mode, which its TFS headers do not identify. "
                         "The native MAD-X Twiss readers require a TWISS table; export TWISS, CHROM instead.")
    if "GAMMA" in headers:
        gamma = float(headers["GAMMA"])
    elif "ENERGY" in headers and "MASS" in headers:
        energy, mass = float(headers["ENERGY"]), float(headers["MASS"])
        if not np.isfinite(energy) or not np.isfinite(mass) or mass <= 0:
            raise ValueError("MAD-X TFS ENERGY and MASS must be finite, with MASS > 0.")
        gamma = energy / mass
    else:
        warnings.warn(
            "MAD-X TFS has neither GAMMA nor ENERGY/MASS; assuming beta0=1 for legacy optics. "
            "Supply reference-energy headers to convert PT dispersion and chromaticity to momentum deviation.",
            RuntimeWarning,
            stacklevel=3)
        return 1.0
    if not np.isfinite(gamma) or gamma <= 1:
        raise ValueError("MAD-X TFS reference GAMMA must be finite and greater than one.")
    return float(np.sqrt(((gamma - 1) / gamma) * ((gamma + 1) / gamma)))


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


def _read_tfs_headers(twiss_file: str | tfs.TfsDataFrame) -> dict:
    """Read raw MADX TFS headers and first-row optics, retaining PT derivatives."""
    df = read_madx_table(twiss_file)
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
        start = s_val - drift_len

        j = i + 1
        while j < len(items):
            if (mergeable(items[j]) and _drifts_are_contiguous(s_val, items[j].s - items[j].length)
                    and _drifts_are_contiguous(start, items[j].s - (drift_len + items[j].length))):
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
            merged._madx_bpm_sources = tuple(dict.fromkeys(source for k in drift_indices for source in getattr(items[k], "_madx_bpm_sources", ())))
            merged._madx_merged_count = sum(getattr(items[k], "_madx_merged_count", 1) for k in drift_indices)
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
        while j < len(items) and str(keywords[j]).casefold() == "drift" and _drifts_are_contiguous(items[j - 1].s, items[j].s_previous):
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
            last._madx_merged_count = sum(getattr(item, "_madx_merged_count", 1) for item in items[i:j])
            result_names.append("_".join(names[i:j]))
        else:
            result_names.append(names[i])
        result_items.append(last)
        i = j
    print(f"[Read MADX Twiss] Merged drifts: {len(items)} -> {len(result_items)}")
    return result_items, result_names


def read_madx_elements(
    twiss_file: str | tfs.TfsDataFrame,
    error_file: str = "",
    is_merge_drift: bool = False,
    is_field_error: bool = False,
    is_alignment_error: bool = False,
    bpm_monitor: dict | None = None,
    strict_geometry: bool = False,
    bpm_names: list[str] | None = None,
    bpm_patterns: list[str] | None = None,
) -> tuple[list, list[str], float]:
    """Read MADX elements → (element_items, element_names, circumference).

    Each MADX element is converted to the corresponding PASS Element schema
    object with its physical parameters (length, strength, edge angles, etc.).

    Naming: f"{madx_name}_s{s:.3f}" (e.g. "qd1_s1.200").

    SBend K0L is automatically patched from the ANGLE column when K0L is
    zero or absent (MADX twiss TFS stores the bend angle in ANGLE, not K0L).

    Args:
        twiss_file: MADX TFS path or the sequence loader's elements table.
        error_file: path to MADX error TFS file.
        is_merge_drift: merge consecutive drift elements.
        is_field_error: attach field errors to matching elements.
        is_alignment_error: attach DX/DY/DPSI to magnetic elements; aperture stays fixed.
        bpm_monitor: optional monitor template, e.g. {"type": "StatMonitor"}.
            None converts finite BPMs to drifts and zero-length BPMs to markers.
            Otherwise a monitor replaces the marker or splits the drift in half.
        strict_geometry: reject inconsistent source and converted geometry (1e-6 m length tolerance).
        bpm_names: exact original names, case-insensitive. None with no patterns selects all.
        bpm_patterns: case-insensitive regex searches, combined with bpm_names by union.
            Explicit empty selectors select none. Unselected BPMs remain passive.

    Returns:
        (items, names, circumference) where items is a list of Element
        schema objects and names is a list of corresponding string names.
    """
    twiss_table = read_madx_table(twiss_file)
    if twiss_table.headers.get("PASS_SOURCE") == "MADX_SEQUENCE":
        if "PASS_UNSUPPORTED" not in twiss_table.columns:
            raise ValueError(
                "Sequence element conversion requires loaded.elements, which retains physical source parameters; use loaded.twiss for Twiss points")
        for _, row in twiss_table.iterrows():
            if row.get("PASS_UNSUPPORTED", ""):
                raise ValueError(f"Element {row['NAME']}: {row['PASS_UNSUPPORTED']}")
    if strict_geometry:
        _validate_madx_geometry(twiss_table, element_mode=True)
    selected_rows = _select_bpm_rows(twiss_table, bpm_monitor, bpm_names, bpm_patterns)
    bpm_items, bpm_item_names = _create_bpm_items(twiss_table, bpm_monitor, include_drifts=True, bpm_names=bpm_names, bpm_patterns=bpm_patterns)
    bpm_nodes = iter(zip(bpm_items, bpm_item_names))
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
        elif _is_bpm(et):
            n_nodes = 3 if l > 0 and i in selected_rows else 1
            for node_index in range(n_nodes):
                item, name = next(bpm_nodes)
                item._match_key = match_key if node_index == 0 else None
                items.append(item)
                names.append(name)
            continue
        else:
            print(f"[Read MADX Elements] Warning: unsupported {et} '{name}' -> drift")
            item = DriftItem(s=s, length=l)

        items.append(item)
        names.append(name)
        # Store match_key on item for error matching (survives drift merge)
        item._match_key = match_key

    # Match errors before any drift merging; both kinds share source identities.
    if is_field_error or is_alignment_error:
        if not error_file:
            raise ValueError("Magnetic-error import requires an error TFS file")
        error_dict = read_madx_errors(error_file,
                                      element_names=twiss_table["NAME"],
                                      is_field_error=is_field_error,
                                      is_alignment_error=is_alignment_error)
        key_to_item = {item._match_key: item for item in items if item._match_key is not None}
        field_count, alignment_count = 0, 0
        for key, errs in error_dict.items():
            if key not in key_to_item:
                raise ValueError(f"Error element {key!r} was lost while building the element sequence")
            item = key_to_item[key]
            if any(errs.get("knl", [])) or any(errs.get("ksl", [])):
                if "is_field_error" not in type(item).model_fields:
                    raise ValueError(f"Field errors on {key!r} are not supported by {type(item).__name__}")
                item.is_field_error = True
                item.field_error_knl, item.field_error_ksl = errs["knl"], errs["ksl"]
                field_count += 1
            alignment = errs.get("alignment", {})
            if any(alignment.values()):
                if "is_alignment_error" not in type(item).model_fields:
                    raise ValueError(f"Alignment errors on {key!r} are not supported by {type(item).__name__}")
                item.is_alignment_error = True
                item.alignment_dx, item.alignment_dy, item.alignment_dpsi = (alignment[name] for name in ("dx", "dy", "dpsi"))
                alignment_count += 1
        print(f"[Read MADX Elements] {field_count} field errors, {alignment_count} alignment errors attached")

    # Merge drifts
    if is_merge_drift:
        items, names = merge_drift_elements(items, names)

    names = _unique_names(names)

    if strict_geometry:
        _validate_imported_geometry(items, names, circumference, element_mode=True)

    # Circumference check
    length_count = sum(getattr(item, "length", 0.0) for item in items)
    diff = length_count - circumference
    if abs(diff) < 1e-6:
        print(f"[Read MADX Elements] Circumference check passed: {length_count:.6f} m")
    else:
        print(f"[Read MADX Elements] Circumference check FAILED: "
              f"theory={circumference}, actual={length_count}, diff={diff:.6e}")

    return items, names, circumference


# Twiss reader (twiss transfer tracking)


def read_madx_twiss(
    twiss_file: str | tfs.TfsDataFrame,
    error_file: str = "",
    muz: float = 0.0,
    dqx: float | str = "from_file",
    dqy: float | str = "from_file",
    is_field_error: bool = False,
    insert_patterns: list[str] | None = None,
    longitudinal_transfer: str = "off",
    is_merge_drift: bool = False,
    is_alignment_error: bool = False,
    bpm_monitor: dict | None = None,
    strict_geometry: bool = False,
    bpm_names: list[str] | None = None,
    bpm_patterns: list[str] | None = None,
) -> tuple[list, list[str], float]:
    """Read MADX optics → (twiss_items, item_names, circumference).

    Each row becomes a TwissItem with current + previous optical functions.
    The first point has previous = current.

    Optionally inserts thin-lens elements (quad/sext/oct/kicker/multipole)
    matched by *insert_patterns* (regex) alongside the twiss points.

    Args:
        twiss_file: MADX TFS path or the sequence loader's twiss table.
        error_file: path to the MADX error TFS file (for field errors).
        muz: longitudinal tune (default 0.0).
        dqx: dQx/d(delta). Float or "from_file" to convert the native PT derivative.
        dqy: dQy/d(delta). Float or "from_file" to convert the native PT derivative.
        is_field_error: if True, read field errors and attach as multipole elements.
        is_merge_drift: merge consecutive Drift elements if present in the result.
        is_alignment_error: unsupported here; use read_madx_elements.
        insert_patterns: regex patterns to match element names for thin-lens insertion.
        longitudinal_transfer: "off" / "drift" / "matrix".
        bpm_monitor: optional PASS monitor template at selected BPM centers S-L/2.
            None retains only zero-length BPM Markers. Transport remains Twiss.
        strict_geometry: validate source ring coverage and converted transport (1e-6 m length tolerance).
        bpm_names: exact original names, case-insensitive; both selectors None selects all.
        bpm_patterns: case-insensitive regex searches, unioned with bpm_names.
            Explicit empty selectors select none. Unselected BPMs remain passive.

    Returns:
        (items, names, circumference) where items is a list of TwissItem
        and optionally Element objects, and names is the corresponding
        list of string names.
    """
    if is_alignment_error:
        raise ValueError("Alignment errors require read_madx_elements; Twiss transfer import cannot move physical magnetic fields")
    source_table = read_madx_table(twiss_file)
    if strict_geometry:
        _validate_madx_geometry(source_table, element_mode=False)
    selected_rows = _select_bpm_rows(source_table, bpm_monitor, bpm_names, bpm_patterns)
    bpm_items, bpm_item_names = _create_bpm_items(source_table, bpm_monitor, bpm_names=bpm_names, bpm_patterns=bpm_patterns)
    twiss_table = _split_bpm_twiss_table(source_table, selected_rows) if selected_rows else source_table
    headers = twiss_table.headers
    reference_beta = _get_madx_reference_beta(headers)
    num_elem = twiss_table.shape[0]

    circumference = headers["LENGTH"]
    qx = headers["Q1"]
    qy = headers["Q2"]
    # MAD-X differentiates with respect to PT; PASS tracks delta=P/P0-1.
    dqx_file = reference_beta * headers["DQ1"]
    dqy_file = reference_beta * headers["DQ2"]

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
    dx = reference_beta * twiss_table["DX"]
    dpx = reference_beta * twiss_table["DPX"]
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
        error_dict = read_madx_errors(error_file, element_names=source_table["NAME"])
        # A localized error must retain its transport endpoint through merging.
        keywords = ["field_error" if item._match_key in error_dict else keyword for item, keyword in zip(items, keywords)]

    if is_merge_drift:
        items, names = merge_drift_twiss_points(items, names, keywords)

    # --- Insert thin-lens elements ---
    if insert_patterns:
        insert_items, insert_names = _insert_elements(source_table, insert_patterns)
        items.extend(insert_items)
        names.extend(insert_names)
        print(f"[Read MADX Twiss] {len(insert_items)} thin-lens elements inserted")

    # --- Attach field errors ---
    if is_field_error:
        error_items = []
        error_names = []
        source_positions, occurrences = {}, {}
        for _, row in source_table.iterrows():
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

    items.extend(bpm_items)
    names.extend(bpm_item_names)
    names = _unique_names(names)

    if strict_geometry:
        _validate_imported_geometry(items, names, circumference, element_mode=False)

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
    twiss_file: str | tfs.TfsDataFrame,
    num_interp_slice: int,
    error_file: str = "",
    muz: float = 0.0,
    dqx: float | str = "from_file",
    dqy: float | str = "from_file",
    is_field_error: bool = False,
    insert_patterns: list[str] | None = None,
    longitudinal_transfer: str = "off",
    interp_kind: str = "phase_hermite",
    is_alignment_error: bool = False,
    bpm_monitor: dict | None = None,
    strict_geometry: bool = False,
    bpm_names: list[str] | None = None,
    bpm_patterns: list[str] | None = None,
) -> tuple[list, list[str], float]:
    """Resample a full ring using phase-constrained quintic Hermite optics.

    ``twiss_file`` accepts a TFS path or the sequence loader's twiss table.
    ``num_interp_slice`` counts base points including 0 and C: N segments
    require N+1 points. Original rows are replaced; kicks, field errors and
    optical discontinuities add split points. Source phases and their full
    tune are preserved. No extrapolation is allowed. Only
    ``interp_kind='phase_hermite'`` is supported. File DQx/DQy and dispersion
    are converted from MAD-X PT derivatives to PASS momentum derivatives.
    Explicit DQx/DQy values already use the PASS convention and are unchanged.
    Alignment errors require physical element maps; use read_madx_elements.
    Zero-length BPMs retain Markers by default. ``bpm_monitor`` replaces them
    with diagnostics, adding exact center endpoints S-L/2 for finite BPMs.
    ``bpm_names`` (exact) and ``bpm_patterns`` (regex search) select original
    names case-insensitively by union; both None selects all, empty lists none.
    ``strict_geometry`` validates source and output geometry, with a 1e-6 m
    length tolerance and no allowance for reversed position ordering.
    """
    if is_alignment_error:
        raise ValueError("Alignment errors require read_madx_elements; interpolated Twiss import cannot move physical magnetic fields")
    from PASS.para.twiss_interpolation import resample_madx_twiss

    return resample_madx_twiss(twiss_file,
                               num_interp_slice,
                               error_file,
                               muz,
                               dqx,
                               dqy,
                               is_field_error,
                               insert_patterns,
                               longitudinal_transfer,
                               interp_kind,
                               bpm_monitor=bpm_monitor,
                               strict_geometry=strict_geometry,
                               bpm_names=bpm_names,
                               bpm_patterns=bpm_patterns)


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
        raise ValueError(f"Error element {name!r} is {reason} in the source Twiss table; "
                         "use unique MAD-X instance names or an explicit NAME occurrence such as Q[2]")
    return candidates[0]


def read_madx_errors(error_file_path: str, *, element_names=None, is_field_error=True, is_alignment_error=False) -> dict[str, dict]:
    """Read selected absolute field and/or DX/DY/DPSI errors from ESAVE TFS.

    Both kinds share the original Twiss instance identities. Missing components
    are zero; finite nonzero values are never thresholded. Unsupported DS/DPHI/
    DTHETA must be zero when alignment import is requested. Aperture and BPM
    error columns are ignored. The default preserves the field-only return form.
    """
    if not is_field_error and not is_alignment_error:
        return {}
    table = tfs.read(error_file_path)
    if "NAME" not in table.columns:
        raise ValueError("Error TFS must contain a NAME column")
    columns = []
    if is_field_error:
        for column in table.columns:
            match = re.fullmatch(r"K(\d+)(S?)L", str(column))
            if match:
                columns.append((column, int(match[1]), bool(match[2])))
        if not columns:
            raise ValueError("Field-error TFS must contain K<n>L or K<n>SL columns")
    if is_alignment_error and not any(column in table.columns for column in ("DX", "DY", "DPSI", "DS", "DPHI", "DTHETA")):
        raise ValueError("Alignment-error TFS must contain alignment columns such as DX, DY or DPSI")
    names = list(table["NAME"] if element_names is None else element_names)
    catalog = _error_instance_catalog(names)
    result = {}
    for row_index, (_, row) in enumerate(table.iterrows(), start=1):
        name = str(row["NAME"])
        record = {}
        nonzero = False
        if is_field_error:
            n = max(order for _, order, _ in columns) + 1
            knl, ksl = np.zeros(n), np.zeros(n)
            for column, order, skew in columns:
                value = float(row[column])
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite field error at row {row_index}, {name!r}, column {column}")
                (ksl if skew else knl)[order] = value
            orders = np.flatnonzero((knl != 0) | (ksl != 0))
            n = int(orders[-1]) + 1 if len(orders) else 0
            record.update(knl=knl[:n].tolist(), ksl=ksl[:n].tolist())
            nonzero = bool(len(orders))
        if is_alignment_error:
            alignment = {}
            for column in ("DX", "DY", "DPSI", "DS", "DPHI", "DTHETA"):
                value = float(row.get(column, 0.0))
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite alignment error at row {row_index}, {name!r}, column {column}")
                if column in ("DS", "DPHI", "DTHETA"):
                    if value != 0.0:
                        raise ValueError(f"Nonzero alignment {column} on {name!r} is not supported; use DX, DY and DPSI only")
                else:
                    alignment[column.lower()] = value
            record["alignment"] = alignment
            nonzero = nonzero or any(alignment.values())
        if not nonzero:
            continue
        key = _match_error_instance(name, catalog)
        if key in result:
            raise ValueError(f"Duplicate error records for {key!r}; refusing to overwrite or add them")
        result[key] = record
    print(f"[Read MADX Errors] {len(table)} rows, {len(result)} nonzero selected errors")
    return result
