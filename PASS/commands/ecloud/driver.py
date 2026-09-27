"""Project saved longitudinal slices onto a local electron-cloud time line."""

from dataclasses import dataclass
import math

import numpy as np

from PASS.utils.constants import const

from .interaction import _stage_cloud_slice


@dataclass(frozen=True)
class BeamSliceInterval:
    """A prescribed, piecewise-constant beam line charge in physical seconds."""

    time_start: float
    time_end: float
    bunch_id: int
    slice_index: int
    n_real_particles: float
    line_charge: float
    beam_beta: float
    bunch_index: int = 0
    particle_indices: object = None
    charge_per_length: float = 0.0


def _host(values):
    return values.get() if hasattr(values, "get") else np.asarray(values)


def _time_tolerance(first, second, duration):
    # Large absolute clocks must never hide a finite fraction of a short pulse.
    return min(8 * max(math.ulp(float(first)), math.ulp(float(second))), 1e-12 * duration)


def build_slice_intervals(beam, slice_set_name, turn, s, xp, *, include_particles=False):
    """Read user slice intervals without folding, rescaling or re-slicing z.

    Recount surviving source particles using saved memberships. Coupled mode
    also retains their global indices and signed charge per unit slice length.
    """
    p = beam.particles
    if not isinstance(p.z, xp.ndarray):
        raise TypeError("ElectronCloud backend does not match the beam particle arrays")
    events = []
    for bunch_index, bunch in enumerate(beam.bunches):
        slices = bunch.slice_sets.get(slice_set_name)
        if slices is None or slices.slice_id is None or slices.slice_table is None:
            raise ValueError("ElectronCloud dynamic mode requires an explicitly executed Slicer")
        if slices.coordinate != "z_rel" or slices.purpose != "general" or getattr(slices, "frame", "lab") != "lab":
            raise ValueError("ElectronCloud dynamic mode requires general, laboratory-frame z_rel slices")
        if getattr(bunch, "collision_frame", None) is not None:
            raise ValueError("ElectronCloud dynamic mode cannot execute in a collision frame")
        if slices.valid_turn != turn or slices.valid_s != s:
            raise ValueError("ElectronCloud dynamic mode requires a Slicer at the current turn and position")
        beta, t0 = float(bunch.beta), float(bunch.t0)
        if not (math.isfinite(beta) and 0 < beta <= 1 and math.isfinite(t0)):
            raise ValueError("ElectronCloud beam reference time and speed must be finite and physical")
        if getattr(slices, "reference_time", None) != t0 or getattr(slices, "reference_beta", None) != beta:
            raise ValueError("ElectronCloud beam reference changed after slicing; explicitly execute Slicer again")
        ratio, charge = float(bunch.ratio), float(bunch.num_charge)
        if not (math.isfinite(ratio) and ratio >= 0 and math.isfinite(charge) and charge != 0):
            raise ValueError("ElectronCloud beam particle weight and signed charge must be finite and physical")
        start, end = bunch.start_idx, bunch.end_idx
        alive = p.tag[start:end] > 0
        ids = xp.asarray(slices.slice_id)
        n_slices = slices.num_slices
        if ids.shape != (end - start, ) or ids.dtype.kind not in "iu":
            raise ValueError("ElectronCloud saved slice memberships do not match the bunch particle range")
        live_ids, live_z = ids[alive], p.z[start:end][alive]
        if not bool(xp.all((live_ids >= 0) & (live_ids < n_slices))) or not bool(xp.all(xp.isfinite(live_z))):
            raise ValueError("ElectronCloud live source particles require valid memberships and finite z")
        table = slices.slice_table
        lower, upper, widths = (_host(table[name]).astype(np.float64) for name in ("z_min", "z_max", "delta_z"))
        if any(values.shape != (n_slices, ) or not np.all(np.isfinite(values)) for values in (lower, upper, widths)):
            raise ValueError("ElectronCloud saved slice intervals must be finite one-dimensional arrays")
        if np.any(upper < lower) or not np.allclose(widths, upper - lower, rtol=8 * np.finfo(float).eps, atol=0):
            raise ValueError("ElectronCloud saved slice widths do not match the saved endpoints")
        low, high = xp.asarray(lower)[live_ids], xp.asarray(upper)[live_ids]
        scale = xp.maximum(xp.maximum(xp.abs(low), xp.abs(high)), xp.abs(live_z))
        tolerance = 8 * np.finfo(p.z.dtype).eps * scale
        if not bool(xp.all((live_z >= low - tolerance) & (live_z <= high + tolerance))):
            raise ValueError("ElectronCloud live source particles lie outside their saved slices; execute a covering Slicer again")
        if live_ids.size:
            counts = _host(xp.bincount(live_ids.astype(xp.int64), minlength=n_slices))
        else:
            counts = np.zeros(n_slices, dtype=np.int64)
        for slice_index in range(n_slices):
            width = float(widths[slice_index])
            n_real = float(counts[slice_index]) * ratio
            if width <= 0:
                if counts[slice_index] > 0:
                    raise ValueError("ElectronCloud occupied slices require strictly positive longitudinal widths")
                continue
            time_start = t0 - float(upper[slice_index]) / (beta * const.c)
            time_end = t0 - float(lower[slice_index]) / (beta * const.c)
            line_charge = n_real * charge * const.e / width
            if not all(math.isfinite(value) for value in (time_start, time_end, n_real, line_charge)) or time_end <= time_start:
                raise ValueError("ElectronCloud slice times and line charge must be finite and temporally resolvable")
            expected_duration = width / (beta * const.c)
            if abs((time_end - time_start) - expected_duration) > 1e-6 * expected_duration:
                raise ValueError("ElectronCloud absolute time cannot resolve the saved slice duration to 1e-6 relative accuracy")
            indices = xp.flatnonzero(alive & (ids == slice_index)) + start if include_particles else None
            events.append(
                BeamSliceInterval(time_start, time_end, int(bunch.bunch_id), slice_index, n_real, line_charge, beta, bunch_index, indices,
                                  ratio * charge * const.e / width))
    events.sort(key=lambda event: (event.time_start, event.time_end, event.bunch_id, event.slice_index))
    if not events:
        raise ValueError("ElectronCloud dynamic mode requires at least one finite slice observation interval")
    for previous, current in zip(events, events[1:]):
        duration = min(current.time_end - current.time_start, previous.time_end - previous.time_start)
        if current.time_start < previous.time_end - _time_tolerance(current.time_start, previous.time_end, duration):
            raise ValueError("ElectronCloud dynamic mode does not support overlapping bunch or slice time intervals")
    return events


def evolve_buildup(cloud, events, turn, *, beam=None, interaction_length=0.0):
    """Advance detached physical state; the command commits after output succeeds."""
    if cloud.last_turn is not None and turn <= cloud.last_turn:
        raise ValueError("ElectronCloud dynamic mode already processed this turn or a later turn")
    if cloud.time is not None:
        first = events[0].time_start
        duration = events[0].time_end - first
        if first < cloud.time - _time_tolerance(first, cloud.time, duration):
            raise ValueError("ElectronCloud dynamic mode cannot rewind physical time; check actual transport and bunch intervals")
    if cloud.configuration.mode == "coupled" and (beam is None or not math.isfinite(interaction_length) or interaction_length < 0):
        raise ValueError("coupled electron clouds require a beam and a finite nonnegative interaction length")
    candidate = cloud.clone()
    history, staged = [], []

    def advance(end_time, event=None):
        before = candidate.n_electrons
        time_start = candidate.time
        primary = candidate.inject_primary(event.n_real_particles)["primary_electrons"] if event is not None else 0.0
        beam_field, witness = None, None
        if candidate.pic is not None and event is not None:
            if event.particle_indices is None:
                raise ValueError("coupled electron clouds require actual saved beam slice memberships")
            indices = event.particle_indices
            p = beam.particles
            witness = (p.x[indices], p.y[indices])
            beam_field = candidate.pic.solve(*witness, event.charge_per_length)
        diagnostics = candidate.advance_to(end_time,
                                           line_charge=event.line_charge if event is not None else 0.0,
                                           beam_beta=event.beam_beta if event is not None else 0.0,
                                           beam_field=beam_field,
                                           witness=witness)
        kick_summary = {}
        if witness is not None and event.particle_indices.size and interaction_length > 0:

            def sample(x, y):
                return diagnostics["mean_ex"], diagnostics["mean_ey"]

            update, kick_summary = _stage_cloud_slice(beam.particles, beam.bunches[event.bunch_index], event.particle_indices, sample,
                                                      interaction_length, candidate.xp)
            kick_summary.pop("bunch_id")
            staged.append(update)
        history.append(
            dict(phase="slice" if event is not None else "gap",
                 bunch_id=event.bunch_id if event is not None else None,
                 slice_index=event.slice_index if event is not None else None,
                 time_start=time_start,
                 time_end=end_time,
                 n_real_particles=event.n_real_particles if event is not None else 0.0,
                 line_charge=event.line_charge if event is not None else 0.0,
                 primary_electrons=primary,
                 n_electrons_before=before,
                 n_electrons_after=candidate.n_electrons,
                 incident_electrons=diagnostics["incident_electrons"],
                 emitted_electrons=diagnostics["emitted_electrons"],
                 steps=diagnostics["steps"],
                 **kick_summary))

    try:
        if candidate.time is None:
            candidate.advance_to(events[0].time_start)
        for event in events:
            if event.time_start > candidate.time:
                advance(event.time_start)
            if event.time_end <= candidate.time:
                raise ValueError("ElectronCloud slice interval is not later than its current time")
            advance(event.time_end, event)
        candidate.set_last_turn(turn)
    except Exception:
        candidate.close()
        raise
    diagnostics = dict(turn=turn,
                       mode=candidate.configuration.mode,
                       time=candidate.time,
                       n_macroparticles=candidate.n_macroparticles,
                       n_electrons=candidate.n_electrons,
                       history=history)
    if candidate.pic is not None:
        diagnostics["max_kick"] = max((row.get("max_kick", 0.0) for row in history), default=0.0)
        diagnostics["max_electric_field"] = max((row.get("max_electric_field", 0.0) for row in history), default=0.0)
    return candidate, diagnostics, staged
