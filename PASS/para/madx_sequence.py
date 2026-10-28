"""Evaluate a MAD-X lattice with optional periodic optics in an isolated process.

The returned tables retain MAD-X PT derivatives. PASS readers perform the
existing conversion to momentum deviation once. Tracking maps are unchanged.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import tfs

from PASS.para.schema.main import MainConfig
from PASS.tool.particle_masses import tracking_mass_per_nucleon


@dataclass
class MadxSequenceData:
    """Detached snapshots; no live MAD-X process or temporary files are retained."""

    elements: tfs.TfsDataFrame
    twiss: tfs.TfsDataFrame | None
    beam: dict
    warnings: tuple[str, ...]
    madx_version: str


def _pass_beam(main, bunch):
    if main is None or bunch is None:
        raise ValueError("beam_source='pass' requires main and one explicitly selected reference bunch")
    main = main if isinstance(main, MainConfig) else MainConfig.model_validate(main)
    if hasattr(bunch, "kinetic_energy"):
        kinetic_energy = bunch.kinetic_energy
    elif isinstance(bunch, dict):
        kinetic_energy = bunch.get("kinetic_energy", bunch.get("Kinetic Energy per Nucleon (eV/u)"))
    else:
        raise ValueError("Select one reference bunch, not a list of bunches")
    if kinetic_energy is None or not np.isfinite(float(kinetic_energy)) or float(kinetic_energy) <= 0:
        raise ValueError("The reference bunch requires finite, positive kinetic_energy (eV/u)")
    mass_number = max(1, main.num_proton + main.num_neutron)
    mass = tracking_mass_per_nucleon(main.num_proton, main.num_neutron, main.num_electron) * mass_number / 1e9
    return dict(particle="ion", mass=mass, charge=main.num_electron, energy=mass + float(kinetic_energy) * mass_number / 1e9)


def _element_limitations(element, attributes):
    """Describe physics which the current element reader would otherwise drop."""
    kind = element.base_type.name.lower()
    supported = {
        "marker", "drift", "sbend", "rbend", "quadrupole", "sextupole", "octupole", "multipole", "solenoid", "hkicker", "vkicker", "kicker",
        "tkicker", "monitor", "hmonitor", "vmonitor"
    }
    limitations = []
    if kind not in supported:
        limitations.append(f"unsupported element type {kind}")

    def nonzero(keys):
        return [key.upper() for key in keys if np.any(np.asarray(attributes.get(key, 0), dtype=float) != 0)]

    omitted = nonzero(("tilt", "dx", "dy", "ds", "dtheta", "dphi", "dpsi", "ktap", "aperture"))
    if omitted:
        limitations.append("unsupported attributes " + ", ".join(omitted))
    if kind in ("quadrupole", "sextupole", "octupole", "multipole", "solenoid"):
        fringe = nonzero(("e1", "e2", "h1", "h2", "hgap", "fint"))
        if kind != "quadrupole":
            fringe.extend(nonzero(("f1", "f2")))
        if kind == "solenoid":
            fringe.extend(nonzero(("fringe", "kill_ent_fringe", "kill_exi_fringe")))
        if kind == "multipole" and int(attributes.get("fringe", 0)) & 6:
            limitations.append("MAD-X MULTIPOLE is a thin map without PTC straight-magnet fringe attributes; use an explicit PASS thick Multipole")
        if attributes.get("fintx", -1) > 0:
            fringe.append("FINTX")
        if fringe:
            limitations.append("unsupported fringe attributes " + ", ".join(fringe))
    if kind in ("sbend", "rbend"):
        omitted = nonzero(
            ("k0s", "k1", "k1s", "k2", "k2s", "k3", "k3s", "knl", "ksl", "h1", "h2", "f1", "f2", "kill_ent_fringe", "kill_exi_fringe", "add_angle"))
        if omitted:
            limitations.append("unsupported bend attributes " + ", ".join(omitted))
        k0, angle, length = float(attributes.get("k0", 0)), float(attributes.get("angle", 0)), float(element.length)
        if k0 and not np.isclose(k0 * length, angle, rtol=1e-12, atol=1e-14):
            limitations.append("independent bend K0 differs from ANGLE/L")
        if attributes.get("fint", 0) != 0 and attributes.get("fintx", -1) == 0:
            limitations.append("explicit FINTX=0 with FINT!=0 is not represented by the existing bend reader")
    if kind in ("quadrupole", "sextupole", "octupole") and nonzero(("knl", "ksl", "k0")):
        limitations.append("additional multipole components are not represented by this element reader")
    if kind == "multipole" and (len(attributes.get("knl", [])) > 21 or len(attributes.get("ksl", [])) > 21):
        limitations.append("multipole orders above 20 are unsupported")
    return "; ".join(limitations)


def _extract_elements(sequence, headers):
    rows = []
    occurrences = {}
    for element, start in zip(sequence.expanded_elements, sequence.expanded_element_positions()):
        attributes = dict(element)
        length = float(element.length)
        name = element.name
        occurrences[name] = occurrences.get(name, 0) + 1
        row = {key.upper(): value for key, value in attributes.items() if isinstance(value, (int, float, bool))}
        row.update(NAME=f"{name}:{occurrences[name]}", KEYWORD=element.base_type.name.upper(), S=float(start) + length, L=length)
        if rows and row["S"] < rows[-1]["S"]:
            # Expanded positions can round a shared exit/marker boundary down by one ULP.
            tolerance = 4 * np.spacing(max(1.0, abs(row["S"]), abs(rows[-1]["S"])))
            if rows[-1]["S"] - row["S"] <= tolerance:
                row["S"] = rows[-1]["S"]
        row["PASS_UNSUPPORTED"] = _element_limitations(element, attributes)
        for order in range(21):
            for skew, array_name in (("", "knl"), ("s", "ksl")):
                values = attributes.get(array_name, [])
                integrated = float(values[order]) if order < len(values) else 0.0
                row[f"K{order}{skew.upper()}L"] = integrated + float(attributes.get(f"k{order}{skew}", 0)) * length
        if row["KEYWORD"] in ("SBEND", "RBEND") and not attributes.get("k0", 0):
            row["K0L"] = attributes.get("angle", 0.0)
        if row["KEYWORD"] == "RBEND":
            row["E1"] = attributes.get("e1", 0) + attributes.get("angle", 0) / 2
            row["E2"] = attributes.get("e2", 0) + attributes.get("angle", 0) / 2
        # MAD-X uses KS when nonzero, and otherwise falls back to integrated KSI.
        row["KSI"] = attributes.get("ks", 0) * length if attributes.get("ks", 0) else attributes.get("ksi", 0)
        # Single-plane kicker maps prefer KICK; the TWISS HKICK/VKICK columns
        # can retain a conflicting alias and are not the effective map strength.
        row["HKICK"] = (attributes.get("kick", 0) if row["KEYWORD"] == "HKICKER" else 0) or attributes.get("hkick", 0)
        row["VKICK"] = (attributes.get("kick", 0) if row["KEYWORD"] == "VKICKER" else 0) or attributes.get("vkick", 0)
        if any(not np.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
            raise ValueError(f"Element {name} has nonfinite parameters")
        rows.append(row)
    table = tfs.TfsDataFrame(rows, headers=headers).fillna(0)
    return table


def _reject_source_errors(madx, sequence_name):
    # Reissuing USE clears node errors. Do not silently erase source EALIGN/EFCOMP.
    if sequence_name not in madx.sequence or not madx.sequence[sequence_name].is_expanded:
        return
    for element in madx.sequence[sequence_name].expanded_elements:
        if any(getattr(element, attribute, 0) != 0 for attribute in ("chkick", "cvkick")):
            raise ValueError(f"Source CORRECT kicks on {element.name} are not supported by sequence loading yet; "
                             "USE would clear these kicks. Encode the correction in the lattice kick parameters.")
        for attribute in ("align_errors", "field_errors", "phase_errors"):
            values = getattr(element, attribute, None)
            if values is not None and any(np.any(np.asarray(value) != 0) for value in values):
                raise ValueError(f"Source {attribute} on {element.name} are not supported by sequence loading yet; "
                                 "USE would clear these errors. Supply an ideal lattice and use the existing error-TFS import.")


def _snapshot_optics(madx, method, elements, beam):
    if method == "twiss":
        madx.command.select(flag="twiss", clear=True)
        raw = madx.twiss(chrom=True, deltap="0", centre=False)
        summary = madx.table.summ.dframe().iloc[0].to_dict()
    else:
        madx.command.ptc_create_universe()
        madx.command.ptc_create_layout(model=2, method=6, nst=1, exact=True, time=True)
        madx.command.ptc_setswitch(time=True, fringe=True)
        madx.command.ptc_twiss(icase=5, no=2, closed_orbit=True)
        raw = madx.table.ptc_twiss
        summary = madx.table.ptc_twiss_summary.dframe().iloc[0].to_dict()
    frame = raw.dframe().copy()
    frame.columns = [column.upper() for column in frame.columns]
    frame.reset_index(drop=True, inplace=True)
    headers = {key.upper(): value for key, value in raw.summary.items()}
    headers.update({key.upper(): value for key, value in summary.items()})
    headers.update({key: value for key, value in beam.items() if isinstance(value, (str, int, float, bool))})
    headers["LENGTH"] = elements.headers["LENGTH"]
    headers["SEQUENCE"] = elements.headers["SEQUENCE"]
    headers.update(NAME="PTC_TWISS" if method == "ptc" else "TWISS",
                   PASS_SOURCE="MADX_SEQUENCE",
                   PASS_DERIVATIVE="PT",
                   PASS_PTC_TIME=True if method == "ptc" else False)
    if method == "ptc":
        for target, source in {
                "BETX": "BETA11",
                "ALFX": "ALFA11",
                "BETY": "BETA22",
                "ALFY": "ALFA22",
                "MUX": "MU1",
                "MUY": "MU2",
                "DX": "DISP1",
                "DPX": "DISP2",
                "DY": "DISP3",
                "DPY": "DISP4"
        }.items():
            frame[target] = frame[source]
        headers["GAMMATR"] = headers["GAMMA_TR"]
        coupling = ("BETA12", "BETA21")
    else:
        coupling = ("R11", "R12", "R21", "R22")
    for column in ("X", "PX", "Y", "PY", "DY", "DPY", *coupling):
        if column in frame and np.max(np.abs(frame[column].to_numpy(dtype=float))) > 1e-10:
            raise ValueError(f"Twiss point import cannot represent nonzero {column} (closed orbit, vertical dispersion or coupling)")
    required = ("S", "BETX", "ALFX", "BETY", "ALFY", "MUX", "MUY", "DX", "DPX")
    if frame.empty or not np.all(np.isfinite(frame[list(required)].to_numpy(dtype=float))):
        raise ValueError("MAD-X did not produce finite periodic optics")
    if np.any(frame[["BETX", "BETY"]].to_numpy() <= 0):
        raise ValueError("MAD-X periodic beta functions must be positive")
    for phase, tune in (("MUX", "Q1"), ("MUY", "Q2")):
        frame[phase] -= frame[phase].iloc[0]
        headers[tune] = float(frame[phase].iloc[-1])
    # Match complete expanded instances, including repeated names and thin nodes.
    if len(frame) != len(elements) or not np.allclose(frame["S"], elements["S"], rtol=0, atol=1e-9):
        raise ValueError("Optics rows do not match the complete expanded sequence; remove source interpolation/range selections")
    frame["NAME"] = elements["NAME"].to_numpy()
    frame["KEYWORD"] = elements["KEYWORD"].to_numpy()
    frame["S"] = elements["S"].to_numpy()
    frame["L"] = elements["L"].to_numpy()
    if method == "ptc":
        madx.command.ptc_end()
    return tfs.TfsDataFrame(frame, headers=headers)


def load_madx_sequence(sequence_file, *, sequence_name="ring", beam_source="file", main=None, bunch=None, optics_method=None):
    """Load a LINE/SEQUENCE and optionally compute uncoupled periodic optics.

    ``beam_source='pass'`` uses MainConfig and one BunchConfig (or field/alias
    dictionaries). ``optics_method`` is None, ``twiss`` or ``ptc``. PTC always
    uses TIME=TRUE. Files execute as MAD-X programs; prefer lattice-only inputs.
    Source CALL paths are relative to the entry file; generated optics run in a
    temporary directory. No matching or tracking-map changes are performed.
    """
    if beam_source not in ("file", "pass"):
        raise ValueError("beam_source must be 'file' or 'pass'")
    if optics_method not in (None, "twiss", "ptc"):
        raise ValueError("optics_method must be None, 'twiss' or 'ptc'")
    if not isinstance(sequence_name, str) or not re.fullmatch(r"[a-zA-Z_][\w.]*", sequence_name):
        raise ValueError("sequence_name must be a MAD-X identifier")
    path = Path(sequence_file).expanduser().resolve(strict=True)
    beam_parameters = _pass_beam(main, bunch) if beam_source == "pass" else None
    try:
        from cpymad.madx import Madx
    except ImportError as exc:
        raise ImportError("cpymad is a required PASS dependency but is unavailable; reinstall PASS with its dependencies") from exc
    messages = []

    def collect_output(chunk):
        text = chunk.decode(errors="replace") if isinstance(chunk, bytes) else str(chunk)
        for line in text.splitlines():
            if re.search(r"number of (warnings|errors):\s*0\s*$", line, re.IGNORECASE):
                continue
            if any(token in line.lower() for token in ("warning", "error", "fatal")):
                messages.append(line.strip())

    try:
        with TemporaryDirectory(prefix="pass_madx_") as scratch, Madx(stdout=collect_output) as madx:
            madx.chdir(scratch)
            if beam_parameters is not None:
                madx.command.beam(**beam_parameters)
            madx.call(str(path), chdir=True)
            _reject_source_errors(madx, sequence_name)
            if beam_parameters is not None:
                madx.command.beam(**beam_parameters)
                if sequence_name in madx.sequence:
                    madx.command.beam(sequence=sequence_name, **beam_parameters)
            madx.use(sequence=sequence_name)
            if sequence_name not in madx.sequence:
                raise ValueError(f"MAD-X sequence {sequence_name!r} was not found")
            sequence = madx.sequence[sequence_name]
            if not sequence.has_beam:
                raise ValueError(f"MAD-X sequence {sequence_name!r} has no BEAM; select beam_source='pass' or define BEAM in the file")
            beam = {key.upper(): value for key, value in dict(sequence.beam).items()}
            if beam["GAMMA"] <= 1 or not np.isfinite(beam["GAMMA"]):
                raise ValueError("MAD-X reference BEAM requires finite GAMMA > 1")
            if beam.get("BV", 1) != 1 or beam.get("DELTAP", 0) != 0 or beam.get("RADIATE", False):
                raise ValueError("Sequence import currently requires BEAM BV=1, DELTAP=0 and RADIATE=FALSE")
            headers = {key: value for key, value in beam.items() if isinstance(value, (str, int, float, bool))}
            headers.update(LENGTH=float(sequence.length), SEQUENCE=sequence_name, PASS_SOURCE="MADX_SEQUENCE", NAME="SEQUENCE")
            elements = _extract_elements(sequence, headers)
            if optics_method == "twiss":
                quadrupoles = elements.loc[elements["KEYWORD"] == "QUADRUPOLE"]
                if any(quadrupoles.get(column, np.array([])).any() for column in ("F1", "F2", "FRINGE")):
                    messages.append("Native TWISS does not reproduce the PTC quadrupole fringe model; "
                                    "choose optics_method='ptc' for fringe_rebuild optics.")
            twiss = _snapshot_optics(madx, optics_method, elements, beam) if optics_method else None
            if twiss is not None:
                for column in ("BETX", "ALFX", "BETY", "ALFY", "MUX", "MUY", "DX", "DPX"):
                    elements[column] = twiss[column].to_numpy(copy=True)
                elements.headers.update(twiss.headers)
            version = str(madx.version)
    except (RuntimeError, KeyError) as exc:
        detail = "; ".join(messages[-5:])
        raise ValueError(f"MAD-X could not load/compute sequence {sequence_name!r}; check the lattice and BEAM definition. {detail}") from exc
    return MadxSequenceData(elements, twiss, beam, tuple(dict.fromkeys(messages)), version)
