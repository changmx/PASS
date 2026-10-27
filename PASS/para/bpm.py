"""Preserve MAD-X BPM locations and optionally attach PASS diagnostics."""

from copy import deepcopy
import math
import re

import tfs

from PASS.para.schema.elements import DriftItem, MarkerItem
from PASS.para.schema.monitors import DistMonitorItem, ParticleMonitorItem, PhaseAdvanceMonitorItem, StatMonitorItem


def _is_bpm(keyword) -> bool:
    return str(keyword).casefold() in ("monitor", "hmonitor", "vmonitor")


def _select_bpm_rows(table, bpm_monitor=None, bpm_names=None, bpm_patterns=None):
    """Select original row positions; omitted selectors mean all BPMs."""
    for label, values in (("bpm_names", bpm_names), ("bpm_patterns", bpm_patterns)):
        if values is not None and (not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values)):
            raise ValueError(f"{label} must be None or a list of nonempty strings")
    names = {name.casefold() for name in bpm_names or []}
    patterns = []
    for pattern in bpm_patterns or []:
        try:
            patterns.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(f"Invalid bpm_patterns expression {pattern!r}: {exc}") from exc
    select_all = bpm_names is None and bpm_patterns is None
    selected_rows = set()
    if bpm_monitor is None:
        return selected_rows
    for index, (_, row) in enumerate(table.iterrows()):
        if not _is_bpm(row.get("KEYWORD", "")):
            continue
        name = str(row["NAME"])
        if select_all or name.casefold() in names or any(pattern.search(name) for pattern in patterns):
            selected_rows.add(index)
    return selected_rows


def _unique_names(names: list[str]) -> list[str]:
    """Keep all source occurrences, including names rounded to the same S."""
    reserved = set(names)
    used = set()
    result = []
    for name in names:
        candidate, suffix = name, 2
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
            while candidate in reserved:
                candidate = f"{name}_{suffix}"
                suffix += 1
        result.append(candidate)
        used.add(candidate)
    return result


def _parse_bpm_monitor(bpm_monitor):
    """Resolve a monitor template without allowing it to move the source BPM."""
    if bpm_monitor is None:
        return None, {}
    if not isinstance(bpm_monitor, dict):
        raise ValueError("bpm_monitor must be None or a dictionary with a 'type' key")
    monitor_types = {
        "stat": StatMonitorItem,
        "statmonitor": StatMonitorItem,
        "dist": DistMonitorItem,
        "distmonitor": DistMonitorItem,
        "particle": ParticleMonitorItem,
        "particlemonitor": ParticleMonitorItem,
        "phaseadvance": PhaseAdvanceMonitorItem,
        "phaseadvancemonitor": PhaseAdvanceMonitorItem,
    }
    monitor_type = str(bpm_monitor.get("type", "")).casefold()
    cls = monitor_types.get(monitor_type)
    if cls is None:
        raise ValueError("bpm_monitor.type must select StatMonitor, DistMonitor, ParticleMonitor or PhaseAdvanceMonitor")
    fields = {key: key for key in cls.model_fields}
    fields.update({field.alias: key for key, field in cls.model_fields.items() if field.alias})
    kwargs = {}
    for key, value in bpm_monitor.items():
        if key == "type":
            continue
        field = fields.get(key)
        if field is None:
            raise ValueError(f"Unknown {cls.__name__} bpm_monitor field: {key}")
        if field in ("s", "command", "order"):
            raise ValueError(f"bpm_monitor cannot override {key}; position and command ordering come from the imported lattice")
        if field in kwargs:
            raise ValueError(f"Duplicate bpm_monitor field: {field}")
        kwargs[field] = deepcopy(value)
    # Validate required user options even when the source has no BPM rows.
    probe = dict(kwargs, s=0.0)
    if cls is PhaseAdvanceMonitorItem:
        for field, value in (("beta_x", 1.0), ("beta_y", 1.0), ("alpha_x", 0.0), ("alpha_y", 0.0)):
            probe.setdefault(field, value)
    cls(**probe)
    return cls, kwargs


def _bpm_center_row(row):
    """Back-propagate exit optics through half of a field-free BPM body."""
    result = row.copy()
    half_length = float(row.get("L", 0.0)) / 2
    result["S"] = float(row["S"]) - half_length
    if half_length == 0:
        return result
    for beta_key, alpha_key, mu_key in (("BETX", "ALFX", "MUX"), ("BETY", "ALFY", "MUY")):
        if beta_key not in row or alpha_key not in row:
            result = result.drop([beta_key, alpha_key, mu_key], errors="ignore")
            continue
        beta, alpha = float(row[beta_key]), float(row[alpha_key])
        if not math.isfinite(beta) or beta <= 0 or not math.isfinite(alpha):
            raise ValueError(f"BPM {row['NAME']}: finite positive {beta_key} and finite {alpha_key} are required")
        gamma = (1 + alpha * alpha) / beta
        result[beta_key] = beta + 2 * half_length * alpha + half_length**2 * gamma
        result[alpha_key] = alpha + half_length * gamma
        if mu_key in row:
            result[mu_key] = float(row[mu_key]) - math.atan2(half_length, beta + half_length * alpha) / (2 * math.pi)
    if "DX" in row:
        result["DX"] = float(row["DX"]) - half_length * float(row.get("DPX", 0.0))
    return result


def _split_bpm_twiss_table(table, selected_rows=None):
    """Add the exact drift midpoint of finite BPMs without changing source exits."""
    rows = []
    source_names = set(table["NAME"])
    for index, (_, row) in enumerate(table.iterrows()):
        if (selected_rows is None or index in selected_rows) and _is_bpm(row.get("KEYWORD", "")) and float(row.get("L", 0.0)) > 0:
            center = _bpm_center_row(row)
            if rows and float(rows[-1]["S"]) > center["S"]:
                raise ValueError(f"BPM {row['NAME']}: its center precedes the previous TFS row; export BPM entrance/exit optics")
            name = f"__pass_bpm_center_{index}"
            while name in source_names:
                name += "_"
            source_names.add(name)
            center["NAME"], center["KEYWORD"], center["L"] = name, "MARKER", 0.0
            rows.append(center)
        rows.append(row)
    return tfs.TfsDataFrame(rows, columns=table.columns, headers=table.headers.copy()).reset_index(drop=True)


def _create_bpm_items(table, bpm_monitor=None, *, include_drifts=False, bpm_names=None, bpm_patterns=None):
    """Convert BPMs to passive bodies or diagnostics at the body center."""
    cls, kwargs = _parse_bpm_monitor(bpm_monitor)
    selected_rows = _select_bpm_rows(table, bpm_monitor, bpm_names, bpm_patterns)
    items, names = [], []
    reference_beta = None
    for index, (_, row) in enumerate(table.iterrows()):
        if not _is_bpm(row.get("KEYWORD", "")):
            continue
        length, s = float(row.get("L", 0.0)), float(row["S"])
        if not math.isfinite(length) or length < 0:
            raise ValueError(f"BPM {row['NAME']}: length must be finite and nonnegative")
        first_item = len(items)
        if index not in selected_rows:
            if length == 0:
                items.append(MarkerItem(s=s))
                names.append(f"{row['NAME']}_s{s:.3f}")
            elif include_drifts:
                items.append(DriftItem(s=s, length=length))
                names.append(f"{row['NAME']}_s{s:.3f}")
            for item in items[first_item:]:
                item._madx_bpm_sources = (index, )
            continue
        sample = _bpm_center_row(row) if cls is PhaseAdvanceMonitorItem else row
        center = s - length / 2
        name = f"{row['NAME']}_s{center:.3f}"
        if include_drifts and length > 0:
            items.append(DriftItem(s=center, length=length / 2))
            names.append(f"{name}_drift_in")
        values = deepcopy(kwargs)
        if cls is PhaseAdvanceMonitorItem:
            for field, column in (("beta_x", "BETX"), ("beta_y", "BETY"), ("alpha_x", "ALFX"), ("alpha_y", "ALFY")):
                if field not in values:
                    if column not in sample:
                        raise ValueError(f"BPM {name}: export {column} or supply bpm_monitor.{field} for PhaseAdvanceMonitor")
                    values[field] = float(sample[column])
            for field, column in (("dx", "DX"), ("dpx", "DPX")):
                if field not in values and column in sample:
                    if reference_beta is None:
                        from PASS.para.madx import _get_madx_reference_beta
                        reference_beta = _get_madx_reference_beta(table.headers)
                    # Native TWISS differentiates dispersion with respect to PT.
                    values[field] = reference_beta * float(sample[column])
            # Source closed orbits are not affine offsets in PASS Twiss maps.
            # Keep the monitor's zero defaults unless explicitly configured.
        items.append(cls(s=center, **values))
        names.append(name)
        if include_drifts and length > 0:
            items.append(DriftItem(s=s, length=length / 2))
            names.append(f"{row['NAME']}_s{s:.3f}_drift_out")
        for item in items[first_item:]:
            item._madx_bpm_sources = (index, )
    return items, names


def describe_madx_bpms(twiss_file, items, names, *, bpm_monitor=None, bpm_names=None, bpm_patterns=None, element_mode=True):
    """Describe each source BPM using final imported command names and geometry.

    Call with the same monitor options used for import. Private source identifiers
    survive drift merging but are deliberately absent from serialized commands.
    """
    from PASS.para.madx import read_madx_table

    if len(items) != len(names):
        raise ValueError("Imported items and names must have equal lengths")
    table = read_madx_table(twiss_file)
    cls, _ = _parse_bpm_monitor(bpm_monitor)
    selected_rows = _select_bpm_rows(table, bpm_monitor, bpm_names, bpm_patterns)
    occurrences, result = {}, []
    for index, (_, row) in enumerate(table.iterrows()):
        source_name = str(row["NAME"])
        occurrences[source_name] = occurrences.get(source_name, 0) + 1
        if not _is_bpm(row.get("KEYWORD", "")):
            continue
        length, source_s = float(row.get("L", 0.0)), float(row["S"])
        selected = index in selected_rows
        outputs = []
        for item, name in zip(items, names):
            command = item.command
            output_length = float(item.s - item.s_previous) if command == "Twiss" else float(getattr(item, "length", 0.0))
            associated = index in getattr(item, "_madx_bpm_sources", ())
            if not element_mode and command == "Twiss" and length > 0:
                associated = min(float(item.s), source_s) > max(float(item.s_previous), source_s - length)
            if associated:
                merged = getattr(item, "_madx_merged_count", 1) > 1
                outputs.append({"name": name, "command": command, "s": float(item.s), "length": output_length, "merged": bool(merged)})
        outputs.sort(key=lambda output: (output["s"], 0 if output["length"] > 0 else 1))
        body = "Drift" if element_mode else "Twiss"
        if selected:
            monitor = cls.model_fields["command"].default
            conversion = f"{body} → {monitor} → {body}" if length > 0 else monitor
        else:
            conversion = body if length > 0 else "Marker"
        result.append({
            "source_name": source_name,
            "occurrence": occurrences[source_name],
            "keyword": str(row["KEYWORD"]).upper(),
            "length": length,
            "source_s": source_s,
            "selected": selected,
            "measurement_s": source_s - length / 2 if selected else None,
            "conversion": conversion,
            "outputs": outputs,
        })
    return result
