"""Versioned, self-describing TFS tables for integrated SI wake responses.

One component occupies one file. Decimal round-trip serialization preserves the
original samples, including the causal right-hand zero limit before half weight.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import shlex

import numpy as np


def _component_term(component, spatial):
    from .wake_components import COMPONENTS, SpatialTerm
    if component not in COMPONENTS and component != "custom":
        raise ValueError(f"Unknown wake component {component!r}")
    if (component == "custom") != (spatial is not None):
        raise ValueError("Only custom components require explicit spatial powers")
    if spatial is not None and not isinstance(spatial, SpatialTerm):
        raise TypeError("Custom wake components require a SpatialTerm")
    return spatial if spatial is not None else SpatialTerm(*COMPONENTS[component])


def _canonical_convention(kind, term, reference_beta):
    from .wake_io import WakeConvention
    order = sum(term.source_powers) + sum(term.test_powers)
    unit = "V/C" if kind == "wake_function" else "ohm"
    if order:
        unit += "/m" if order == 1 else f"/m^{order}"
    return WakeConvention(data_kind=kind,
                          axis="time" if kind == "wake_function" else "frequency",
                          axis_unit="s" if kind == "wake_function" else "Hz",
                          value_unit=unit,
                          positive_trailing=True,
                          longitudinal_positive_loss=True,
                          fourier_exponent=-1,
                          transverse_impedance_factor="i",
                          shunt_impedance_convention="not_applicable",
                          integrated=True,
                          reference_beta=reference_beta)


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON metadata value {value}")


def _parse_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite JSON metadata value {value}")
    return number


def _parse_json(value, key):
    try:
        return json.loads(value, parse_constant=_reject_json_constant, parse_float=_parse_json_float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid JSON in wake TFS {key}") from exc


def _parse_tfs(payload):
    """Read the strict numeric TFS subset without rounding adjacent float nodes."""
    headers, names, types, rows = {}, None, None, []
    for number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(("#", "!")):
            continue
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError as exc:
            raise ValueError(f"Invalid quoting in wake TFS line {number}") from exc
        if not tokens:
            continue
        marker = tokens[0]
        if marker == "@":
            if names is not None or len(tokens) != 4:
                raise ValueError(f"Invalid wake TFS header at line {number}")
            _, key, dtype, value = tokens
            if key in headers:
                raise ValueError(f"Duplicate wake TFS header {key}")
            if re.fullmatch(r"%\d*s", dtype):
                category = "string"
            elif dtype in {"%d", "%hd", "%ld"}:
                category = "integer"
                try:
                    value = int(value)
                except ValueError as exc:
                    raise ValueError(f"Wake TFS {key} must be an integer") from exc
            elif dtype in {"%le", "%lf", "%e", "%f"}:
                category = "real"
                try:
                    value = float(value)
                except ValueError as exc:
                    raise ValueError(f"Wake TFS {key} must be a real number") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Wake TFS {key} must be finite")
            else:
                raise ValueError(f"Unsupported wake TFS header type {dtype}")
            headers[key] = (category, value)
        elif marker == "*":
            if names is not None or types is not None or len(tokens) < 2:
                raise ValueError("Wake TFS requires one column declaration")
            names = tokens[1:]
            if len(set(names)) != len(names):
                raise ValueError("Wake TFS column names must be unique")
        elif marker == "$":
            if names is None or types is not None or len(tokens) != len(names) + 1:
                raise ValueError("Invalid wake TFS column types")
            types = tokens[1:]
            if any(dtype not in {"%le", "%lf", "%e", "%f"} for dtype in types):
                raise ValueError("Wake TFS data columns must have real floating types")
        else:
            if names is None or types is None or len(tokens) != len(names):
                raise ValueError(f"Invalid wake TFS data row at line {number}")
            try:
                rows.append([float(value) for value in tokens])
            except ValueError as exc:
                raise ValueError(f"Non-numeric wake TFS data at line {number}") from exc
    if names is None or types is None or len(rows) < 2:
        raise ValueError("Wake TFS requires column/type declarations and at least two rows")
    table = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(table)):
        raise ValueError("Wake TFS data must be finite")
    return headers, names, table


def _header(headers, key, category):
    if key not in headers:
        raise ValueError(f"Missing required wake TFS header {key}")
    found_category, value = headers[key]
    if found_category != category:
        raise ValueError(f"Wake TFS {key} must have {category} type")
    return value


def _read_spatial_term(headers):
    from .wake_components import SpatialTerm
    powers = []
    for key in ("SOURCE_POWERS", "TEST_POWERS"):
        value = _parse_json(_header(headers, key, "string"), key)
        if not isinstance(value, list):
            raise ValueError(f"Wake TFS {key} must be a JSON list")
        powers.append(value)
    return SpatialTerm(_header(headers, "PLANE", "string"), *powers)


def _read_contract(headers, component, spatial):
    term = _component_term(component, spatial)
    if _header(headers, "PASS_WAKE_VERSION", "integer") != 1:
        raise ValueError("Unsupported PASS_WAKE_VERSION; expected 1")
    kind = _header(headers, "DATA_KIND", "string")
    if kind not in {"wake_function", "impedance"}:
        raise ValueError("Wake TFS DATA_KIND must be wake_function or impedance; wake potentials require deconvolution")
    if _header(headers, "COMPONENT", "string") != component:
        raise ValueError("Wake TFS COMPONENT conflicts with configured component")
    if _read_spatial_term(headers) != term:
        raise ValueError("Wake TFS plane/spatial powers conflict with configured component")
    reference_beta = _header(headers, "REFERENCE_BETA", "real")
    convention = _canonical_convention(kind, term, reference_beta)
    strings = {
        "AXIS_UNIT": convention.axis_unit,
        "VALUE_UNIT": convention.value_unit,
        "TRANSVERSE_IMPEDANCE_FACTOR": convention.transverse_impedance_factor,
        "SHUNT_IMPEDANCE_CONVENTION": convention.shunt_impedance_convention,
    }
    for key, expected in strings.items():
        if _header(headers, key, "string") != expected:
            raise ValueError(f"Wake TFS {key} must use canonical value {expected!r}")
    for key, expected in (("INTEGRATED", 1), ("POSITIVE_TRAILING", 1), ("LONGITUDINAL_POSITIVE_LOSS", 1), ("FOURIER_EXPONENT", -1)):
        if _header(headers, key, "integer") != expected:
            raise ValueError(f"Wake TFS {key} must be {expected}")
    return convention


def read_wake_tfs(path, *, component=None, spatial=None, convention=None, causal=None, reconstruction="two_sided"):
    """Read canonical metadata; reject conflicting caller units or conventions.

    Temporal tables define their own causality. Spectrum reconstruction remains
    an explicit caller choice because finite bandwidth alone is not causal.
    """
    from .wake_io import WakeConvention
    from .wake_models import TabulatedWakeModel
    from .wake_spectrum import ImpedanceSpectrum, SpectrumWakeModel
    path = Path(path)
    payload = path.read_bytes()
    headers, names, table = _parse_tfs(payload)
    if component is None:
        component = _header(headers, "COMPONENT", "string")
    if component == "custom" and spatial is None:
        spatial = _read_spatial_term(headers)
    file_convention = _read_contract(headers, component, spatial)
    if convention is not None:
        if not isinstance(convention, WakeConvention):
            raise TypeError("Supply a WakeConvention when overriding wake TFS metadata")
        for key, value in asdict(file_convention).items():
            if getattr(convention, key) != value:
                raise ValueError(f"Wake TFS convention conflicts with configured {key}")
    if causal is not None and type(causal) is not bool:
        raise ValueError("Wake TFS causal override must be a boolean")
    source_metadata = None
    if "SOURCE_METADATA" in headers:
        source_metadata = _parse_json(_header(headers, "SOURCE_METADATA", "string"), "SOURCE_METADATA")
        if not isinstance(source_metadata, dict):
            raise ValueError("Wake TFS SOURCE_METADATA must be a JSON object")
    if file_convention.data_kind == "wake_function":
        if names != ["TAU", "W"]:
            raise ValueError("Wake-function TFS columns must be exactly TAU W")
        causal_value = _header(headers, "CAUSAL", "integer")
        if causal_value not in {0, 1}:
            raise ValueError("Wake TFS CAUSAL must be 0 or 1")
        file_causal = bool(causal_value)
        zero_value = _header(headers, "ZERO_VALUE", "string")
        if zero_value != ("right_limit" if file_causal else "sample"):
            raise ValueError("Wake TFS ZERO_VALUE conflicts with its causality")
        if causal is not None and causal != file_causal:
            raise ValueError("Wake TFS CAUSAL conflicts with configured causal flag")
        model = TabulatedWakeModel(table[:, 0], table[:, 1], causal=file_causal)
    else:
        if names != ["FREQUENCY", "REAL", "IMAG"]:
            raise ValueError("Impedance TFS columns must be exactly FREQUENCY REAL IMAG")
        if "CAUSAL" in headers or "ZERO_VALUE" in headers:
            raise ValueError("Impedance TFS must not declare temporal CAUSAL or ZERO_VALUE")
        longitudinal = _component_term(component, spatial).plane == "z"
        model = SpectrumWakeModel(ImpedanceSpectrum(*table.T, longitudinal=longitudinal), reconstruction)
        if causal is not None and causal != model.causal:
            raise ValueError("Wake TFS causal override conflicts with spectrum reconstruction")
        zero_value = "right_limit" if model.causal else "sample"
    model.input_metadata = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "format": "tfs",
        "component": component,
        "spatial": asdict(_component_term(component, spatial)),
        "convention": asdict(file_convention),
        "columns": list(range(len(names))),
        "column_names": names,
        "source_metadata": source_metadata,
        "causal": model.causal,
        "zero_value": zero_value,
    }
    return model


def _check_destination(path, source_metadata):
    metadata = source_metadata
    while isinstance(metadata, dict):
        source = metadata.get("path")
        if source is not None:
            source = Path(source)
            if source.resolve() == path.resolve() or (source.exists() and path.exists() and source.samefile(path)):
                raise ValueError("Wake TFS destination must not overwrite its source file")
        metadata = metadata.get("source_metadata")


def write_wake_tfs(path, model, *, component, spatial=None, reference_beta, overwrite=False):
    """Write exact temporal/spectrum samples; never fit, resample or halve zero.

    Validate and serialize before opening the target. Existing files require
    explicit overwrite; source files are always protected, including hard links.
    """
    from .wake_models import TabulatedWakeModel
    from .wake_spectrum import ImpedanceSpectrum, SpectrumWakeModel
    if type(overwrite) is not bool:
        raise ValueError("Wake TFS overwrite must be a boolean")
    term = _component_term(component, spatial)
    if isinstance(model, TabulatedWakeModel):
        if type(model.causal) is not bool:
            raise ValueError("Tabulated wake causality must be a boolean")
        checked = TabulatedWakeModel(model.times, model.values, causal=model.causal)
        kind, names = "wake_function", ["TAU", "W"]
        table = np.column_stack((checked.times, checked.values))
    elif isinstance(model, SpectrumWakeModel):
        spectrum = model.spectrum
        if spectrum.longitudinal != (term.plane == "z"):
            raise ValueError("Spectrum plane conflicts with configured wake component")
        checked = ImpedanceSpectrum(spectrum.frequencies, spectrum.real, spectrum.imag, spectrum.longitudinal)
        kind, names = "impedance", ["FREQUENCY", "REAL", "IMAG"]
        table = np.column_stack((checked.frequencies, checked.real, checked.imag))
    else:
        raise TypeError("Wake TFS serialization requires a TabulatedWakeModel or SpectrumWakeModel")
    convention = _canonical_convention(kind, term, reference_beta)
    headers = {
        "PASS_WAKE_VERSION": 1,
        "DATA_KIND": kind,
        "COMPONENT": component,
        "PLANE": term.plane,
        "SOURCE_POWERS": json.dumps(list(term.source_powers)),
        "TEST_POWERS": json.dumps(list(term.test_powers)),
        "AXIS_UNIT": convention.axis_unit,
        "VALUE_UNIT": convention.value_unit,
        "REFERENCE_BETA": float(reference_beta),
        "INTEGRATED": 1,
        "POSITIVE_TRAILING": 1,
        "LONGITUDINAL_POSITIVE_LOSS": 1,
        "FOURIER_EXPONENT": -1,
        "TRANSVERSE_IMPEDANCE_FACTOR": "i",
        "SHUNT_IMPEDANCE_CONVENTION": "not_applicable",
    }
    if kind == "wake_function":
        headers.update(CAUSAL=int(model.causal), ZERO_VALUE="right_limit" if model.causal else "sample")
    path = Path(path)
    source_metadata = getattr(model, "input_metadata", None)
    if source_metadata is not None:
        if not isinstance(source_metadata, dict):
            raise ValueError("Wake input_metadata must be a JSON object")
        headers["SOURCE_METADATA"] = json.dumps(source_metadata, ensure_ascii=True, allow_nan=False, sort_keys=True)
        _check_destination(path, source_metadata)
    lines = []
    for key, value in headers.items():
        if isinstance(value, str):
            dtype, rendered = "%s", json.dumps(value, ensure_ascii=False)
        elif isinstance(value, int):
            dtype, rendered = "%d", str(value)
        else:
            dtype, rendered = "%le", format(value, ".17g")
        lines.append(f"@ {key} {dtype} {rendered}")
    lines.extend(("* " + " ".join(names), "$ " + " ".join(["%le"] * len(names))))
    lines.extend(" ".join(format(float(value), ".17g") for value in row) for row in table)
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    with path.open("wb" if overwrite else "xb") as stream:
        stream.write(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(table),
        "data_kind": kind,
        "component": component,
        "columns": names,
        "reference_beta": float(reference_beta),
    }
