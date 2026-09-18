"""Translate the actual-magnet BRing CISP case into a reviewable PASS input.

This is a deliberately scoped importer, not a generic CISP parser. Unknown
active elements fail. Source files are read only; every conversion is recorded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import numpy as np
import tfs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PASS.para.api import generate_input
from PASS.para.schema import MainConfig, Sequence, StatMonitorItem, DistMonitorItem, SlicerItem
from PASS.para.schema.bunch import BunchConfig, InjectionItem, OffsetConfig
from PASS.para.schema.elements import (DriftItem, SBendItem, QuadrupoleItem, MultipoleItem, SextupoleItem, BumpItem, RFCavityItem, ElSeparatorItem)
from PASS.para.schema.space_charge import SpaceChargeConfig, SpaceChargeResourceConfig, ElementSpaceCharge
from PASS.utils.bump_waveform import convert_cisp_bump
from PASS.utils.constants import const


def number(v):
    value, *kind = str(v).split("|")
    if kind and kind != ["value"]:
        raise ValueError(f"Expected a constant, got {v}")
    return float(value)


def source_slices(length, maximum, explicit=1):
    """CISP Element::sliceElement default, including its even slice count."""
    if explicit > 1:
        return explicit
    return 2 * (math.floor(length / (2 * maximum)) + 1) if length > maximum else 1


def dipole_fringe_integrals(name, values):
    """Translate independent CISP integrals to the legacy PASS Fint/Fintx pair."""
    entrance, exit_ = number(values["fint1"]), number(values["fint2"])
    if not all(math.isfinite(value) and value >= 0 for value in (entrance, exit_)):
        raise ValueError(f"Dipole {name}: fint1 and fint2 must be finite and non-negative")
    if entrance > 0 and exit_ == 0:
        raise ValueError(f"Dipole {name}: fint1={entrance:g}, fint2={exit_:g} cannot be represented "
                         "by PASS Fint/Fintx: Fintx <= 0 inherits the entrance Fint. "
                         "A nonzero entrance integral with an independently zero exit integral "
                         "requires a different source magnet representation.")
    return entrance, exit_


def parse(path):
    definitions, order = {}, None
    for statement in path.read_text(encoding="utf-8").split(";"):
        statement = statement.strip()
        if not statement:
            continue
        kind, body = statement.split(":", 1)
        kind = kind.strip().lower()
        if kind == "beamline":
            order = [item.strip() for item in body.split(",")]
            continue
        values = dict(part.strip().split("=", 1) for part in body.split(",") if "=" in part)
        values = {key.strip(): value.strip() for key, value in values.items()}
        definitions[values.get("name", kind)] = (kind, values)
    if order is None:
        raise ValueError("Missing CISP beamline")
    return definitions, order


def build(source,
          output,
          *,
          turns=100,
          particles_per_batch=None,
          stage="aperture",
          clock="reference",
          backend="gpu",
          rf_directory=None,
          snapshots=True,
          grid_width=.5,
          grid_cells=None,
          es_length=1.,
          particle_clock_origin="cisp",
          rf_voltage_unit=None,
          es_voltage=None,
          es_gap=None,
          es_vl=None):
    if turns < 1 or not np.isfinite(grid_width) or grid_width <= 0:
        raise ValueError("turns and grid_width must be positive")
    if rf_voltage_unit is not None and rf_voltage_unit not in {"V", "MV"}:
        raise ValueError("rf_voltage_unit must be V or MV")
    if rf_directory is not None and rf_voltage_unit is None:
        raise ValueError("RF input requires an explicit rf_voltage_unit (V or MV)")
    if stage not in {"external", "aperture", "pic"}:
        raise ValueError("Unknown comparison stage")
    if particle_clock_origin not in {"cisp", "injection"}:
        raise ValueError("particle_clock_origin must be cisp or injection")
    physical_es = stage != "external"
    if physical_es:
        if (es_voltage is None) == (es_vl is None) or es_gap is None:
            raise ValueError(
                "The physical ES requires exactly one of es_voltage (V) and es_vl (V m), plus es_gap (m); source septum cuts do not specify these values"
            )
        strength = es_voltage if es_voltage is not None else es_vl
        if not all(np.isfinite(v) for v in (strength, es_gap)) or es_gap <= 0:
            raise ValueError("ES strength must be finite; gap must be positive and finite")
    source, output = Path(source).resolve(), Path(output).resolve()
    definitions, order = parse(source)
    # Validate all active dipoles before writing any converted files.
    dipole_integrals = {}
    for name in order:
        kind, values = definitions[name]
        if kind == "thickdipole" and values.get("switch", "on") != "off":
            dipole_integrals[name] = dipole_fringe_integrals(name, values)
    output.mkdir(parents=True, exist_ok=True)
    base = source.parent.parent
    provenance = {}

    def dependency(value):
        path = Path(value.split("|")[0])
        path = path if path.is_absolute() else base / path
        if not path.is_file():
            raise FileNotFoundError(path)
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    provenance[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    ring = definitions["ring"][1]
    circumference = number(ring["circumference"])
    source_length = math.fsum(number(definitions[name][1].get("length", "0")) for name in order)
    if abs(source_length - circumference) > 1e-7:
        raise ValueError(f"Source ring length {source_length} != declared circumference {circumference}")
    if physical_es:
        available = sum(number(definitions[name][1]["length"]) for name in ("drift049a", "drift049b"))
        if not np.isfinite(es_length) or not 0 < es_length <= available:
            raise ValueError(f"es_length must lie in (0, {available}] m")
    # Source lengths and the rounded circumference differ by about 12 nm.
    # Split actual lengths against their sum, without manufacturing a tiny
    # extra ES segment at a source boundary. Final S is rounded below.
    es_start = source_length - es_length
    injector_name = next(name for name in order if definitions[name][0] == "injector")
    injection_index = order.index(injector_name)
    inj = definitions[injector_name][1]
    if any(number(inj[f"bump_{coordinate}_{axis}"]) != 0 for coordinate in ("pos", "mom") for axis in "xy"):
        raise ValueError("Importer requires actual-magnet painting (Injector equivalent bumps must be zero)")
    batch_count = int(inj["inj_turns"])
    per_batch = int(inj["particles_per_turn"]) if particles_per_batch is None else particles_per_batch
    if not 1 <= per_batch <= int(inj["particles_per_turn"]):
        raise ValueError("particles_per_batch must be within the source batch size")
    incoming = np.loadtxt(dependency(inj["file"]), delimiter=",", ndmin=2)[:per_batch].copy()
    if incoming.shape != (per_batch, 6):
        raise ValueError("Invalid incoming six-dimensional CSV")
    # CISP first makes a LINEAR energy offset from the CSV delta. Recover
    # the corresponding physical momentum for PASS rather than relabeling it.
    energy = number(inj["ke"]) * 1e6
    mass = 931.49410242e6
    gamma = 1 + energy / mass
    beta2 = 1 - 1 / gamma**2
    de = (incoming[:, 5] + number(inj["ddp"])) * beta2 * (energy + mass)
    ratio = (2 * (energy + mass) * de + de * de) / (energy * (energy + 2 * mass))
    incoming[:, 5] = ratio / (np.sqrt(1 + ratio) + 1)
    distribution = output / "incoming_batch.tfs"
    tfs.write(distribution, tfs.TfsDataFrame(dict(zip(("x", "px", "y", "py", "z", "dp"), incoming.T))))
    offsets = {
        axis: OffsetConfig(is_offset=True, offset_position=number(inj[f"inj_pos_{axis}"]), offset_momentum=number(inj[f"inj_mom_{axis}"]))
        for axis in "xy"
    }
    bunch = BunchConfig(kinetic_energy=energy,
                        num_macro_particles=per_batch * batch_count,
                        num_real_particles=int(inj["particles_per_turn"]) * batch_count * int(inj["ion_num"]),
                        injection_turns=batch_count * int(inj["inj_interval"]),
                        injection_interval=int(inj["inj_interval"]),
                        is_load_from_file=True,
                        file_path=str(distribution),
                        file_mode="repeat",
                        beta_x=number(inj["beta_x"]),
                        beta_y=number(inj["beta_y"]),
                        alpha_x=number(inj["alpha_x"]),
                        alpha_y=number(inj["alpha_y"]),
                        emit_x=number(inj["emit_x"]),
                        emit_y=number(inj["emit_y"]),
                        sigma_z=float(np.std(incoming[:, 4])),
                        dp=float(np.std(incoming[:, 5])),
                        offset_x=offsets["x"],
                        offset_y=offsets["y"])
    theta = number(inj["es_angle"])
    outer = number(inj["es_x"]) * np.sin(theta) + number(inj["es_y"]) * np.cos(theta)
    thickness = number(inj["es_thickness"])
    seq = Sequence().add("injection", InjectionItem(bunches=[bunch]))
    selected_turns = sorted(set([0, 1] + list(range(4, batch_count, 5)) + [batch_count - 1, batch_count, 72, turns - 1]))
    selected_turns = [[t] for t in selected_turns if 0 <= t < turns]
    if snapshots:
        seq.add("after_injection", DistMonitorItem(s=0., save_turns=selected_turns, include_injection_metadata=True, output_format="hdf5-gzip1"))
    sc_on = stage == "pic"
    resource = definitions["sc01"][1]
    if sc_on:
        seq.add(
            "slices",
            SlicerItem(s=0.,
                       slice_set="injection",
                       num_slices=int(resource["slice_num_t"]),
                       z_range_mode="explicit",
                       explicit={
                           "z min": -number(ring["circumference"]) / 2,
                           "z max": number(ring["circumference"]) / 2
                       },
                       coordinate="z_periodic"))
    rotated = order[injection_index + 1:] + order[:injection_index]
    position, location_map = 0., []
    max_slice = number(ring["max_slice_length"])
    for name in rotated:
        kind, v = definitions[name]
        length = number(v.get("length", "0"))
        position += length
        s = position
        if abs(s - number(ring["circumference"])) < 1e-7:
            s = number(ring["circumference"])
        location_map.append({"name": name, "s": s, "originally_upstream": bool(order.index(name) < injection_index), "first_reference_time_s": 0.0})
        if kind in {"grfuncfree2p5sc", "tunemonitor", "phasemonitor"}:
            continue
        aperture = {"aperture_type": "off"}
        if stage != "external" and v.get("aper_switch") == "on":
            aper_type = v["aper_type"]
            if aper_type == "polygon":
                vertices = np.loadtxt(dependency(v["aper_file"]), delimiter=",", ndmin=2)
                if np.array_equal(vertices[0], vertices[-1]):
                    vertices = vertices[:-1]
                dims = vertices.tolist()
            elif aper_type == "rectangle":
                dims = [number(v["half_width"]), number(v["half_height"])]
            elif aper_type == "circle":
                dims = [number(v["radius"])]
            else:
                raise ValueError(f"Unsupported source aperture {aper_type}")
            aperture = {"aperture_type": aper_type, "aperture_value": dims}
        options = dict(s=s, length=length, **aperture)
        slicing = dict(num_slices=source_slices(length, max_slice, int(v.get("element_slice_num", "1"))))
        if sc_on and length and v.get("sc") == "sc01":
            slicing["space_charge"] = ElementSpaceCharge(configuration="cisp", num_kicks=slicing["num_slices"])
        if v.get("switch", "on") == "off":
            if length:
                seq.add(name, DriftItem(**options, **slicing))
            continue
        if kind == "thickdrift":
            if physical_es and name in {"drift049a", "drift049b"} and s > es_start + 1e-10:
                # The physical ES is the last 1 m before injection. The CISP
                # collimator occupies only the final 0.75 m, so retain its
                # aperture and split the preceding drift when necessary.
                original_start = position - length
                if original_start < es_start - 1e-10:
                    approach_length = es_start - original_start
                    approach_slicing = dict(num_slices=source_slices(approach_length, max_slice))
                    if "space_charge" in slicing:
                        approach_slicing["space_charge"] = ElementSpaceCharge(configuration="cisp", num_kicks=approach_slicing["num_slices"])
                    seq.add(name + "_approach", DriftItem(s=es_start, length=approach_length, **aperture, **approach_slicing))
                    options["length"] = position - es_start
                    slicing["num_slices"] = source_slices(options["length"], max_slice)
                    if "space_charge" in slicing:
                        slicing["space_charge"] = ElementSpaceCharge(configuration="cisp", num_kicks=slicing["num_slices"])
                item = ElSeparatorItem(**options,
                                       **slicing,
                                       voltage=es_voltage,
                                       gap=es_gap,
                                       voltage_length=None if es_vl is None else es_vl * (options["length"] / es_length),
                                       tilt=theta - np.pi / 2,
                                       septum_position=outer - thickness,
                                       septum_thickness=thickness)
            else:
                item = DriftItem(**options, **slicing)
        elif kind == "thickquadrupole":
            item = QuadrupoleItem(**options, **slicing, k1l=number(v["k1"]) * length, model="mat-kick-mat")
        elif kind == "thickdipole":
            # CISP uses matrix body transport. Converge PASS's nonlinear body
            # integration independently of the source SC kick spacing.
            slicing["num_slices"] = max(8, slicing["num_slices"])
            fint, fintx = dipole_integrals[name]
            item = SBendItem(**options,
                             **slicing,
                             integrator="yoshida4",
                             k0l=number(v["angle"]),
                             e1=number(v["e1"]),
                             e2=number(v["e2"]),
                             hgap=number(v["hgap"]),
                             fint=fint,
                             fintx=fintx)
        elif kind == "multipole":
            if stage == "external":
                continue
            item = MultipoleItem(**options,
                                 knl=[float(x) for x in v["knl"].split("|")[0].split("/")],
                                 ksl=[float(x) for x in v["ksl"].split("|")[0].split("/")])
        elif kind == "sextupole":
            item = SextupoleItem(**options, k2l=number(v["k2l"]))
        elif kind == "bump":
            path = output / (name + ".tfs")
            convert_cisp_bump(dependency(v["kick_x"]), dependency(v["kick_y"]), path)
            # Translate CISP's first-passage local clock to the injection clock
            # once. An explicitly global input retains its raw timestamps.
            time_offset = -s / (np.sqrt(beta2) * const.c) if clock == "particle" and particle_clock_origin == "cisp" else 0.
            item = BumpItem(**options, waveform_file=str(path), time_mode=clock, time_offset=time_offset)
            location_map[-1]["waveform_time_offset_s"] = time_offset
        elif kind == "paramonitor":
            item = StatMonitorItem(s=s)
        elif kind == "rf":
            if rf_directory is None:
                continue
            files = [Path(rf_directory) / v[key].split("|")[0] for key in ("voltage", "phi")]
            tables = [np.loadtxt(dependency(str(path)), delimiter=",") for path in files]
            for path, table in zip(files, tables):
                if (table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 2 or not np.all(np.isfinite(table))
                        or np.any(np.diff(table[:, 0]) <= 0)):
                    raise ValueError(f"RF program {path} requires two finite columns and strictly increasing times")
            rows, time, kinetic = [], 0., energy
            for turn in range(turns):
                if any(time < a[0, 0] or time > a[-1, 0] for a in tables):
                    raise ValueError("RF tables do not cover the requested tracking interval")
                voltage, phase = [float(np.interp(time, *a.T)) for a in tables]
                # Current CISP rf.cpp reads volts and converts to MeV/e
                # internally. Historical files may instead contain MV;
                # require that convention to be explicitly selected.
                voltage *= 1. if rf_voltage_unit == "V" else 1e6
                rows.append([time, voltage, phase])
                kinetic += int(inj["e_num"]) / int(inj["a_num"]) * voltage * np.sin(phase)
                beta = np.sqrt(1 - 1 / (1 + kinetic / mass)**2)
                time += number(ring["circumference"]) / (beta * const.c)
            from PASS.para.tools.rf_data import synchronous_rf_program
            harmonic = int(number(v["h"]))
            values = np.asarray(rows)
            program = synchronous_rf_program(values[:, 1],
                                             values[:, 2],
                                             harmonic,
                                             number(ring["circumference"]),
                                             mass,
                                             energy,
                                             int(inj["e_num"]) / int(inj["a_num"]),
                                             origin=s / (np.sqrt(1 - 1 / (1 + energy / mass)**2) * const.c))
            rf_file = output / "rf_physical_time.tfs"
            tfs.write(rf_file, program, colwidth=25, headerswidth=25)
            item = RFCavityItem(s=s, components=[dict(program_file=str(rf_file))])

        else:
            raise ValueError(f"Unsupported active CISP element {name}: {kind}")
        seq.add(name, item)
    circumference = number(ring["circumference"])
    if abs(position - circumference) > 1e-7:
        raise ValueError(f"Translated ring length {position} != source {circumference}")
    main = MainConfig(beam_name="U238_35_painting",
                      num_proton=92,
                      num_neutron=146,
                      num_electron=35,
                      circumference=circumference,
                      gamma_t=number(ring["gamma_t"]),
                      num_turns=turns,
                      backend=backend,
                      output_dir=str(output / "tracking"))
    sc = SpaceChargeConfig(enabled=sc_on,
                           configurations={
                               "cisp":
                               SpaceChargeResourceConfig(slice_set="injection",
                                                         nx=2**int(resource["nx"]) if grid_cells is None else grid_cells,
                                                         ny=2**int(resource["ny"]) if grid_cells is None else grid_cells,
                                                         solver="fft_free_space",
                                                         deposition_method="CIC",
                                                         grid_width_x=grid_width,
                                                         grid_width_y=grid_width)
                           } if sc_on else {})
    input_path = generate_input(main, seq, str(output / "beam0.json"), space_charge=sc)
    manifest = dict(
        source=str(source),
        stage=stage,
        clock=clock,
        turns=turns,
        per_batch=per_batch,
        batches=batch_count,
        total_particles=per_batch * batch_count,
        source_files_sha256=provenance,
        location_map=location_map,
        rf_enabled=rf_directory is not None,
        rf_voltage_source_unit=rf_voltage_unit if rf_directory is not None else None,
        es_geometry_sampling="continuous_segment_intersections",
        particle_clock_origin=particle_clock_origin,
        source_length_m=source_length,
        final_s_rounding_m=circumference - source_length,
        pic_grid_width_m=grid_width,
        pic_grid_cells=sc.configurations["cisp"].nx if sc_on else None,
        source_particles_per_batch=int(inj["particles_per_turn"]),
        total_real_particles=bunch.num_real_particles,
        comparison_limits=[
            "PASS uses its own external-element maps; CISP uses different chromatic/kinematic approximations.",
            "Matching RF voltage/phase does not equate RF maps: the inspected CISP kernel additionally rescales energy deviation during acceleration; validate RF separately.",
            "CISP applies SC before each source slice; PASS schedules internal SC at midpoints. Step-size convergence is required.",
            "The ES uses finite longitudinal length, infinite electrode height and continuous segment-wall intersections; CISP point septum/collimator cuts are a different loss model.",
            f"PIC: CISP adapts its grid to 1.1*max(abs(x,y)); this input uses a fixed {grid_width:g} m domain. Grid convergence is required.",
            "CISP skips upstream element updates before particles exist; every Bump first samples time zero."
        ],
        electrostatic_separator=dict(enabled=physical_es,
                                     length_m=es_length if physical_es else None,
                                     voltage_V=es_voltage,
                                     integrated_voltage_V_m=es_vl,
                                     gap_m=es_gap,
                                     voltage_definition="septum potential minus counter-electrode potential",
                                     length_is_not_tilt_rescaled=True))
    (output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return Path(input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="CISP command file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--particles-per-batch", type=int)
    parser.add_argument("--stage", choices=["external", "aperture", "pic"], default="aperture")
    parser.add_argument("--clock", choices=["reference", "particle"], default="reference")
    parser.add_argument("--particle-clock-origin",
                        choices=["cisp", "injection"],
                        default="cisp",
                        help="Align CISP local waveform origins to first reference passage, or treat raw timestamps as global")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--rf-directory", type=Path)
    parser.add_argument("--rf-voltage-unit",
                        choices=["V", "MV"],
                        help="Required with --rf-directory: current CISP uses V, some historical files use MV")
    parser.add_argument("--es-length", type=float, default=1., help="Supplied ES length before injection, in m; no tilt rescaling")
    es_strength = parser.add_mutually_exclusive_group()
    es_strength.add_argument("--es-voltage",
                             type=float,
                             help="Septum minus high-voltage-electrode potential, V; choose this or --es-vl outside external stage")
    es_strength.add_argument("--es-vl",
                             type=float,
                             help="Total longitudinal integral of the ES voltage difference, V m; distributed by segment length")
    parser.add_argument("--es-gap", type=float, help="Required outside external stage: clear electrode gap, m")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--grid-width", type=float, default=.5, help="Full PIC domain width in m")
    parser.add_argument("--grid-cells", type=int, help="PIC cells per axis; default from source")
    args = parser.parse_args()
    build(args.source,
          args.output,
          turns=args.turns,
          particles_per_batch=args.particles_per_batch,
          stage=args.stage,
          clock=args.clock,
          backend=args.backend,
          rf_directory=args.rf_directory,
          snapshots=not args.no_snapshots,
          rf_voltage_unit=args.rf_voltage_unit,
          es_voltage=args.es_voltage,
          es_gap=args.es_gap,
          es_vl=args.es_vl,
          grid_width=args.grid_width,
          grid_cells=args.grid_cells,
          es_length=args.es_length,
          particle_clock_origin=args.particle_clock_origin)
