"""Wake projection into whole-ring arrival-phase bins.

This is a diagnostic/projection coordinate. The continuous particle coordinate,
arrival correction, particle order and bunch membership are never replaced.
Each snapshot represents one passage per live macro particle per reference turn;
it is not an event scheduler for zero/multiple passages within a turn.
"""
import numpy as np

from PASS.utils.constants import const


def prepare_periodic_slices(p, bunch, slices, turn=None, location=None):
    xp = getattr(p, "xp", np)
    circumference = float(bunch.circum)
    if (not np.isfinite(circumference) or circumference <= 0
            or slices.explicit.z_max != 0.
            or not np.isclose(slices.explicit.z_min, -circumference, rtol=1e-13, atol=0.)):
        raise ValueError("Periodic arrival slicing requires Explicit [-circumference, 0] in metres")
    if not 0 < bunch.beta <= 1:
        raise ValueError("Periodic arrival slicing requires 0 < beta <= 1")
    sl = slice(bunch.start_idx, bunch.end_idx)
    z = p.z[sl].astype(xp.float64, copy=False)
    offset = p.arrival_offset[sl] if hasattr(p, "arrival_offset") else 0.
    phase = (z+bunch.z_center-bunch.beta*const.c*offset)/circumference
    alive = p.tag[sl] > 0
    if not bool(xp.all(xp.isfinite(phase) | ~alive)):
        raise ValueError("Periodic arrival phase must be finite for live particles")
    history = getattr(slices, "_periodic_previous", {})
    baselines = getattr(slices, "_periodic_baselines", {})
    previous = history.get(location)
    step = 0.
    if previous is not None and turn is not None:
        old_turn, old_phase, old_alive = previous
        if turn == old_turn+1:
            baselines[location] = previous
        elif turn != old_turn:
            baselines.pop(location, None)
        # Repeated projections in one turn must use the same previous-turn
        # baseline; otherwise reslicing could bypass the wake slip guard.
        baseline = baselines.get(location)
        if baseline is not None:
            old_turn, old_phase, old_alive = baseline
        if phase.size and old_phase.shape == phase.shape and turn == old_turn+1:
            step = float(xp.max(xp.where(alive & old_alive, xp.abs(phase-old_phase), 0.)))
    # Map to [-C,0): exact endpoints have one owner, so charge is not duplicated.
    coordinate = (xp.remainder(phase, 1.)-1.)*circumference
    slices._periodic_coordinate = xp.where(alive, coordinate, 0.).astype(p.z.dtype)
    if turn is not None:
        history[location] = (turn, phase.copy(), alive.copy())
        slices._periodic_previous = history
        slices._periodic_baselines = baselines
    slices._periodic_sample = (turn, bunch.beta, bunch.t0, bunch.z_center, circumference, phase)
    slices.periodic_max_step = step


def validate_periodic_wake(beam, name, turn):
    """All populations in a coasting snapshot share one reference-time window."""
    sets = [b.slice_sets.get(name) for b in beam.bunches]
    if any(getattr(s, "coordinate", None) == "ring_position" for s in sets):
        raise ValueError("WakeField requires z_rel or arrival_phase slices; ring_position loses passage timing")
    if not any(s is not None and s.periodic for s in sets):
        return
    if not all(s is not None and s.periodic and s.valid_turn == turn for s in sets):
        raise ValueError("A periodic wake requires fresh periodic slices for every bunch population")
    reference = beam.bunches[0]
    p = beam.particles
    xp = getattr(p, "xp", np)
    immediate = getattr(getattr(beam, "wake_clock", None), "_gpu_immediate_geometry", False)
    for b, s in zip(beam.bunches, sets):
        # Local diagnostic projection is well-defined for arbitrary slip.
        # This bound protects the one-passage-per-reference-turn wake model,
        # so enforce it only when a wake actually consumes the snapshot.
        step = getattr(s, "periodic_max_step", 0.)
        if step > s.max_phase_slip:
            raise ValueError(f"Coasting phase slip {step:.6g} turns exceeds Max phase slip "
                             f"{s.max_phase_slip:g}; the reference-turn approximation is unresolved")
        if not (b.beta == reference.beta and b.t0 == reference.t0
                and b.circum == reference.circum):
            raise ValueError("Periodic wake populations must share beta, t0 and circumference")
        if s.valid_s is None:
            raise ValueError("Periodic wake requires a Slicer at the wake location")
        sample = getattr(s, "_periodic_sample", None)
        if sample is None or sample[:5] != (turn, b.beta, b.t0, b.z_center, b.circum):
            raise ValueError("Periodic reference changed since Slicer; reslice at the wake location")
        if not immediate:
            # Thin transverse elements and diagnostics can lie between Slicer
            # and WakeField. Reuse geometry only if the arrival phase is intact.
            sl = slice(b.start_idx, b.end_idx)
            z = p.z[sl].astype(xp.float64, copy=False)
            offset = p.arrival_offset[sl] if hasattr(p, "arrival_offset") else 0.
            phase = (z+b.z_center-b.beta*const.c*offset)/b.circum
            if phase.shape != sample[5].shape or not bool(xp.all((phase == sample[5]) | (p.tag[sl] <= 0))):
                raise ValueError("Periodic arrival phase changed since Slicer; reslice before WakeField")
