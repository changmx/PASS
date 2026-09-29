"""Convert external wake data to canonical, self-describing PASS wake TFS.

The source knots are preserved. Only declared axis, unit and sign conversions
are performed; no time-to-turn conversion, interpolation or fitting occurs.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from PASS.commands.wake.wake_components import COMPONENTS, SpatialTerm
from PASS.commands.wake.wake_io import WakeConvention, read_wake_file
from PASS.commands.wake.wake_models import TabulatedWakeModel
from PASS.commands.wake.wake_spectrum import ImpedanceSpectrum, SpectrumWakeModel
from PASS.para.schema.wake_field import WakeFileConvention, WakeSpatialTerm
from PASS.utils.constants import const


def _value_scale(unit, kind, order):
    """Accept a small explicit unit grammar; never guess dimensional powers."""
    unit = unit.replace(" ", "").replace("(", "").replace(")", "").replace(".", "*")
    unit = unit.replace("**", "^").replace("*", "/")
    parts = unit.split("/")
    if kind == "wake_function":
        charges = [part for part in parts[1:] if part in {"C", "nC", "pC"}]
        if len(parts) < 2 or parts[0] not in {"V", "kV", "MV"} or len(charges) != 1:
            raise ValueError("Wake units must explicitly specify voltage / charge / spatial powers")
        scale = {"V": 1., "kV": 1e3, "MV": 1e6}[parts.pop(0)]
        scale /= {"C": 1., "nC": 1e-9, "pC": 1e-12}[charges[0]]
        parts.remove(charges[0])
    else:
        if parts[0] not in {"ohm", "Ohm", "kOhm", "MOhm"}:
            raise ValueError("Impedance units must specify ohm / spatial powers")
        scale = {"ohm": 1., "Ohm": 1., "kOhm": 1e3, "MOhm": 1e6}[parts.pop(0)]
    degree = 0
    for part in parts:
        match = re.fullmatch(r"(m|cm|mm)(?:\^(\d+))?", part)
        if match is None:
            raise ValueError(f"Unsupported wake unit term {part!r}")
        power = int(match[2] or 1)
        degree += power
        scale /= {"m": 1., "cm": .01, "mm": .001}[match[1]]**power
    if degree != order:
        raise ValueError(f"Wake unit spatial order {degree} does not match required order {order}")
    return scale


def read_external_wake(path,
                       *,
                       convention: WakeConvention | None = None,
                       component: str,
                       spatial=None,
                       format="table",
                       axis_column=0,
                       value_column=None,
                       imag_column=None,
                       delimiter=None,
                       skiprows=0,
                       causal=None,
                       reconstruction="two_sided",
                       length=None):
    """Read external numeric data for conversion, never for runtime tracking.

    HEADTAIL is an explicit ns, V/pC, V/(pC*mm) format contract, with
    user-selected columns because several HEADTAIL table layouts exist.
    A wake potential is not a point-charge wake and needs separate deconvolution.
    """
    if format not in {"table", "headtail"}:
        raise ValueError("External wake conversion accepts table or headtail; use read_wake_file for standard TFS")
    if isinstance(convention, dict):
        if convention.get("data_kind", convention.get("Data kind")) == "wake_potential":
            raise ValueError("Wake-potential input requires explicit deconvolution before importing a point-charge wake")
        convention = WakeConvention(**WakeFileConvention.model_validate(convention).model_dump())
    if isinstance(spatial, dict):
        spatial = SpatialTerm(**WakeSpatialTerm.model_validate(spatial).model_dump())
    value_column = 1 if value_column is None else value_column
    if causal is not None and type(causal) is not bool:
        raise ValueError("Causal must be a boolean")
    file_convention = convention
    if not isinstance(file_convention, WakeConvention):
        raise TypeError("Supply an explicit WakeConvention for file units, signs and normalization")
    if file_convention.data_kind not in {"wake_function", "impedance"}:
        raise ValueError("Wake-potential input requires explicit deconvolution; supply a wake function or impedance")
    if not np.isfinite(file_convention.reference_beta) or not 0 < file_convention.reference_beta <= 1:
        raise ValueError("File reference beta must lie in (0, 1]")
    if file_convention.fourier_exponent not in {-1, 1} or file_convention.transverse_impedance_factor not in {"i", "-i", "1"}:
        raise ValueError("Invalid Fourier convention")
    if (component == "custom") != (spatial is not None):
        raise ValueError("Only custom components require explicit spatial powers")
    if component not in COMPONENTS and component != "custom":
        raise ValueError(f"Unknown wake component {component!r}")
    term = spatial if spatial is not None else SpatialTerm(*COMPONENTS[component])
    order = sum(term.source_powers) + sum(term.test_powers)
    scale = _value_scale(file_convention.value_unit, file_convention.data_kind, order + int(not file_convention.integrated))
    if not file_convention.integrated:
        if length is None or not np.isfinite(length) or length <= 0:
            raise ValueError("A per-length wake/impedance requires positive physical Length (m)")
        scale *= length
    elif length is not None:
        raise ValueError("Integrated wake data must not be multiplied by another length")
    if format == "headtail":
        if (file_convention.data_kind != "wake_function" or file_convention.axis != "time" or file_convention.axis_unit != "ns"
                or not file_convention.integrated or order > 1
                or _value_scale(file_convention.value_unit, file_convention.data_kind, order) != 1e12 * 1e3**order):
            raise ValueError("HEADTAIL requires ns and integrated V/pC or V/(pC*mm) units")
    columns = [axis_column, value_column] + ([] if imag_column is None else [imag_column])
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in columns + [skiprows]) or len(set(columns)) != len(columns):
        raise ValueError("File columns must be distinct nonnegative integers; Skip rows >= 0")
    payload = Path(path).read_bytes()
    table = np.loadtxt(io.StringIO(payload.decode("utf-8-sig")), delimiter=delimiter, skiprows=skiprows, ndmin=2)
    if len(table) < 2 or max(columns) >= table.shape[1] or not np.all(np.isfinite(table[:, columns])):
        raise ValueError("Wake file needs at least two finite rows and the declared columns")
    axis = table[:, axis_column].copy()
    values = table[:, value_column] * scale
    if term.plane == "z" and not file_convention.longitudinal_positive_loss:
        values = -values
    if file_convention.data_kind == "impedance":
        if file_convention.axis != "frequency" or file_convention.axis_unit not in {"Hz", "kHz", "MHz", "GHz"} or imag_column is None:
            raise ValueError("Impedance input requires frequency units and real/imaginary columns")
        if not file_convention.positive_trailing:
            raise ValueError("Frequency data require a positive-trailing delay convention; use Fourier exponent for transform sign")
        axis *= {"Hz": 1., "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[file_convention.axis_unit]
        imag = table[:, imag_column] * scale
        if term.plane == "z" and not file_convention.longitudinal_positive_loss:
            imag = -imag
        transfer = values + 1j * imag
        if term.plane != "z":
            transfer /= {"i": 1j, "-i": -1j, "1": 1.}[file_convention.transverse_impedance_factor]
        if file_convention.fourier_exponent == 1:
            transfer = transfer.conj()
        z = transfer if term.plane == "z" else 1j * transfer
        model = SpectrumWakeModel(ImpedanceSpectrum(axis, z.real, z.imag, term.plane == "z"), reconstruction)
        if causal is not None and causal != model.causal:
            raise ValueError("Explicit Causal conflicts with impedance reconstruction")
    else:
        if imag_column is not None:
            raise ValueError("A real wake function does not take an imaginary column")
        scales = {"time": {"s": 1., "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}, "distance": {"m": 1., "cm": .01, "mm": .001}}
        if file_convention.axis not in scales or file_convention.axis_unit not in scales[file_convention.axis]:
            raise ValueError("Wake function axis must specify time or distance units")
        axis *= scales[file_convention.axis][file_convention.axis_unit]
        if file_convention.axis == "distance":
            axis /= const.c * (file_convention.reference_beta if file_convention.distance_convention == "beta_c_tau" else 1.)
        if not file_convention.positive_trailing:
            axis = -axis
        # Accept either file direction, but reject unordered/duplicate samples.
        if np.all(np.diff(axis) < 0):
            axis, values = axis[::-1], values[::-1]
        model = TabulatedWakeModel(axis, values, causal=True if causal is None else causal)
    model.input_metadata = {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "format": format,
        "convention": asdict(file_convention),
        "columns": columns,
        "component": component,
        "spatial": asdict(term),
        "import_options": {
            "format": format,
            "component": component,
            "spatial": asdict(spatial) if spatial is not None else None,
            "convention": asdict(file_convention),
            "axis_column": axis_column,
            "value_column": value_column,
            "imag_column": imag_column,
            "delimiter": delimiter,
            "skiprows": skiprows,
            "length": length,
            "causal": model.causal,
            "reconstruction": reconstruction,
        }
    }
    return model


def _load_wake(source, options):
    options = dict(options)
    if isinstance(options.get("spatial"), dict):
        options["spatial"] = SpatialTerm(**WakeSpatialTerm.model_validate(options["spatial"]).model_dump())
    reader = read_wake_file if options.get("format") == "tfs" else read_external_wake
    model = reader(source, **options)
    component = model.input_metadata["component"]
    spatial = SpatialTerm(**model.input_metadata["spatial"])
    return model, component, spatial


def _canonical_data(model, component, spatial):
    order = sum(spatial.source_powers) + sum(spatial.test_powers)
    suffix = "" if order == 0 else "/m" if order == 1 else f"/m^{order}"
    convention = dict(model.input_metadata["convention"])
    convention.update(positive_trailing=True,
                      longitudinal_positive_loss=True,
                      integrated=True,
                      fourier_exponent=-1,
                      transverse_impedance_factor="i",
                      shunt_impedance_convention="not_applicable",
                      distance_convention="beta_c_tau")
    notices = []
    if isinstance(model, TabulatedWakeModel):
        columns = {"TAU": model.times, "W": model.values}
        units = {"TAU": "s", "W": "V/C" + suffix}
        convention.update(data_kind="wake_function", axis="time", axis_unit="s", value_unit=units["W"])
        if model.causal:
            notices.append("The zero-delay sample is W(0+); tracking applies the half point-self-wake exactly once.")
        else:
            notices.append("Two-sided wake data require an isolated or periodic spatial boundary, not causal passage history.")
        if model.values[-1] != 0:
            notices.append("The final wake sample is nonzero; the response is zero beyond the table. Check tail-cutoff convergence.")
        if not model.causal and model.values[0] != 0:
            notices.append("The first two-sided sample is nonzero; the response is zero before the table.")
        amplitude = np.abs(model.values)
    elif isinstance(model, SpectrumWakeModel):
        spectrum = model.spectrum
        columns = {"FREQUENCY": spectrum.frequencies, "REAL": spectrum.real, "IMAG": spectrum.imag}
        units = {"FREQUENCY": "Hz", "REAL": "ohm" + suffix, "IMAG": "ohm" + suffix}
        convention.update(data_kind="impedance", axis="frequency", axis_unit="Hz", value_unit=units["REAL"])
        notices.append("Only the supplied frequency band is retained; reconstruction and bandwidth require independent convergence checks.")
        amplitude = np.abs(spectrum.values)
    else:
        raise TypeError("Wake conversion requires tabulated wake data or sampled impedance")
    axis = next(iter(columns.values()))
    spacing = np.diff(axis)
    metadata = {
        "component": component,
        "spatial": asdict(spatial),
        "convention": convention,
        "source": model.input_metadata,
    }
    if isinstance(model, TabulatedWakeModel):
        metadata.update(causal=model.causal, zero_value="right_limit" if model.causal else "sample")
    else:
        metadata.update(causal=model.causal, reconstruction=model.reconstruction)
    diagnostics = {
        "axis_min": float(axis[0]),
        "axis_max": float(axis[-1]),
        "min_spacing": float(spacing.min()),
        "max_spacing": float(spacing.max()),
        "max_abs_value": float(amplitude.max()),
        "tail_relative_amplitude": float(amplitude[-1] / amplitude.max()) if amplitude.max() else 0.,
        "samples_preserved": True,
    }
    return columns, units, metadata, notices, diagnostics


def _preview_indices(columns, limit):
    """Keep endpoints and signed extrema, then retain each bucket's envelope."""
    axis, *values = columns.values()
    n = len(axis)
    if n <= limit:
        return np.arange(n)
    if len(values) == 2:
        values = [*values, np.hypot(*values)]
    mandatory = {0, n - 1}
    for value in values:
        mandatory.update((int(np.argmin(value)), int(np.argmax(value))))
    if len(mandatory) > limit:
        raise ValueError(f"Preview limit must be at least {len(mandatory)} to retain endpoints and global extrema")
    selected = set(mandatory)
    buckets = (limit - len(mandatory)) // (2 * len(values))
    if buckets:
        for bucket in np.array_split(np.arange(n), buckets):
            for value in values:
                selected.update((int(bucket[np.argmin(value[bucket])]), int(bucket[np.argmax(value[bucket])])))
    indices = np.array(sorted(selected), dtype=int)
    if len(indices) < limit:
        unused = np.ones(n, dtype=bool)
        unused[indices] = False
        candidates = np.flatnonzero(unused)
        fill = candidates[np.linspace(0, len(candidates) - 1, limit - len(indices), dtype=int)]
        indices = np.sort(np.r_[indices, fill])
    return indices


def _component_config(path, model, component, spatial):
    """A file component template; the simulation still chooses its solver/history."""
    config = {
        "Component": component,
        "Velocity": {
            "Kind": "fixed",
            "Beta": model.input_metadata["convention"]["reference_beta"]
        },
        "Model": {
            "Kind": "file",
            "Format": "tfs",
            "File path": str(Path(path).resolve())
        },
    }
    if component == "custom":
        config["Spatial"] = WakeSpatialTerm(**asdict(spatial)).model_dump(by_alias=True, mode="json")
    if isinstance(model, SpectrumWakeModel):
        config["Model"]["Reconstruction"] = model.reconstruction
    return config


def preview_wake_file(source, *, preview_limit=2000, **options):
    """Preview converted SI samples and physical diagnostics without writing.

    Numeric import options match :func:`read_external_wake`. For ``format=tfs``,
    the physical contract is read from the file and the component is optional.
    CSV header rows are skipped explicitly using ``skiprows``; column indices start at zero.
    ``sha256`` identifies the actual bytes read, for use at export time.
    """
    if isinstance(preview_limit, bool) or not isinstance(preview_limit, int) or preview_limit < 2:
        raise ValueError("Preview limit must be an integer >= 2")
    model, component, spatial = _load_wake(source, options)
    columns, units, metadata, notices, diagnostics = _canonical_data(model, component, spatial)
    n = len(next(iter(columns.values())))
    indices = _preview_indices(columns, preview_limit)
    if n > preview_limit:
        notices.append(f"Preview displays {len(indices)} of {n} original knots with endpoints and extrema retained; export preserves all {n} knots.")
    return {
        "source": model.input_metadata["path"],
        "sha256": model.input_metadata["sha256"],
        "component": component,
        "data_kind": metadata["convention"]["data_kind"],
        "rows": n,
        "preview_rows": len(indices),
        "columns": {
            key: values[indices].tolist()
            for key, values in columns.items()
        },
        "units": units,
        "metadata": metadata,
        "notices": notices,
        "diagnostics": diagnostics,
        "component_config": _component_config(source, model, component, spatial) if options.get("format") == "tfs" else None,
    }


def convert_wake_file(source, destination, *, expected_sha256=None, overwrite=False, **options):
    """Normalize external data and publish a standard wake TFS atomically.

    Existing files require ``overwrite=True``; the source can never be a target.
    A hash from :func:`preview_wake_file` rejects a source changed since preview.
    Time tables retain W(0+) rather than an already halved point response.
    """
    from PASS.commands.wake.wake_tfs import _check_destination, write_wake_tfs

    if type(overwrite) is not bool:
        raise ValueError("Overwrite must be a boolean")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.suffix.lower() != ".tfs":
        raise ValueError("Standard wake output must have the .tfs extension")
    if source == destination or destination.exists() and os.path.samefile(source, destination):
        raise ValueError("The source file must not be overwritten")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    model, component, spatial = _load_wake(source, options)
    _check_destination(destination, model.input_metadata)
    source_sha256 = model.input_metadata["sha256"]
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise ValueError("Source changed since preview; preview the file again before exporting")
    columns, _, metadata, notices, _ = _canonical_data(model, component, spatial)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pass-wake-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        write_wake_tfs(staged,
                       model,
                       component=component,
                       spatial=spatial if component == "custom" else None,
                       reference_beta=metadata["convention"]["reference_beta"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
            raise ValueError("Source changed during conversion; output was not published")
        if destination.exists() and os.path.samefile(source, destination):
            raise ValueError("The destination aliases the source file")
        if overwrite:
            os.replace(staged, destination)
        else:
            # Exclusive publication also protects files created after the first check.
            os.link(staged, destination)
    return {
        "outputs": [str(destination)],
        "row_counts": [len(next(iter(columns.values())))],
        "notices": notices,
        "sha256": source_sha256,
        "metadata": metadata,
        "component_config": _component_config(destination, model, component, spatial),
    }


def main(argv=None):
    """Command-line wake import; run ``python -m PASS.tool.wake_conversion -h``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination", nargs="?", help="Output .tfs file; omit when using --preview")
    parser.add_argument("--preview", action="store_true", help="Print converted samples and diagnostics without writing")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--component", choices=[*COMPONENTS, "custom"], help="Required for numeric input; inferred from standard TFS")
    parser.add_argument("--plane", choices=["x", "y", "z"])
    parser.add_argument("--source-powers", nargs=2, type=int, default=[0, 0])
    parser.add_argument("--test-powers", nargs=2, type=int, default=[0, 0])
    parser.add_argument("--format", choices=["table", "headtail", "tfs"], default="table")
    parser.add_argument("--data-kind", choices=["wake_function", "impedance"])
    parser.add_argument("--axis", choices=["time", "distance", "frequency"])
    parser.add_argument("--axis-unit")
    parser.add_argument("--distance-convention", choices=["beta_c_tau", "c_tau"], help="Distance coordinate: s=beta*c*tau (default) or s=c*tau")
    parser.add_argument("--value-unit")
    parser.add_argument("--reference-beta", type=float)
    parser.add_argument("--negative-trailing", action="store_true")
    parser.add_argument("--positive-gain", action="store_true", help="Positive input longitudinal values mean energy gain")
    parser.add_argument("--per-length", action="store_true")
    parser.add_argument("--length", type=float)
    parser.add_argument("--fourier-exponent", choices=[-1, 1], type=int)
    parser.add_argument("--transverse-impedance-factor", choices=["i", "-i", "1"])
    parser.add_argument("--axis-column", type=int, default=0)
    parser.add_argument("--value-column", type=int)
    parser.add_argument("--imag-column", type=int)
    parser.add_argument("--delimiter", help="Default: whitespace; use ',' for CSV or 'tab' for a tab")
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--two-sided", action="store_true", help="Input wake table is two-sided")
    parser.add_argument("--reconstruction", choices=["two_sided", "causal_projection"], default="two_sided")
    args = parser.parse_args(argv)
    options = {
        "component": args.component,
        "format": args.format,
        "reconstruction": args.reconstruction,
    }
    if args.component == "custom":
        if args.plane is None and args.format != "tfs":
            parser.error("--component custom requires --plane")
        if args.plane is not None:
            options["spatial"] = dict(plane=args.plane, source_powers=args.source_powers, test_powers=args.test_powers)
        elif args.source_powers != [0, 0] or args.test_powers != [0, 0]:
            parser.error("Explicit custom powers require --plane")
    elif args.plane is not None or args.source_powers != [0, 0] or args.test_powers != [0, 0]:
        parser.error("Spatial powers and --plane apply only to --component custom")
    if args.format != "tfs":
        if any(value is None for value in (args.component, args.axis, args.axis_unit, args.value_unit, args.reference_beta)):
            parser.error("Numeric input requires --component, --axis, --axis-unit, --value-unit and --reference-beta")
        options.update(axis_column=args.axis_column,
                       value_column=args.value_column,
                       imag_column=args.imag_column,
                       delimiter="\t" if args.delimiter == "tab" else args.delimiter,
                       skiprows=args.skiprows,
                       causal=False if args.two_sided else None,
                       length=args.length)
        options["convention"] = dict(data_kind=args.data_kind or "wake_function",
                                     axis=args.axis,
                                     axis_unit=args.axis_unit,
                                     value_unit=args.value_unit,
                                     distance_convention=args.distance_convention or "beta_c_tau",
                                     reference_beta=args.reference_beta,
                                     positive_trailing=not args.negative_trailing,
                                     longitudinal_positive_loss=not args.positive_gain,
                                     integrated=not args.per_length,
                                     fourier_exponent=-1 if args.fourier_exponent is None else args.fourier_exponent,
                                     transverse_impedance_factor=args.transverse_impedance_factor or "i")
    elif (any(value is not None for value in (args.axis, args.axis_unit, args.distance_convention, args.value_unit, args.reference_beta,
                                              args.data_kind, args.fourier_exponent, args.transverse_impedance_factor)) or args.negative_trailing
          or args.positive_gain or args.per_length):
        parser.error("Standard TFS already defines its data kind, axis, units, signs and reference beta")
    elif (args.axis_column != 0 or args.value_column is not None or args.imag_column is not None or args.delimiter is not None or args.skiprows
          or args.length is not None or args.two_sided):
        parser.error("Standard TFS already defines its named SI columns and causality; numeric import options are not applicable")
    if not args.preview and args.destination is None:
        parser.error("Supply a destination or use --preview")
    try:
        result = (preview_wake_file(args.source, **options) if args.preview else convert_wake_file(
            args.source, args.destination, expected_sha256=args.expected_sha256, overwrite=args.overwrite, **options))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
