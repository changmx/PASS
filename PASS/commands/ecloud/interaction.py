"""Reference-speed transverse thin kicks from a prescribed electron cloud."""

import math

from PASS.utils.constants import const


def stage_frozen_cloud(beam, fields, interaction_length):
    """Return diagnostics and validated updates without changing a beam.

    ``brho`` includes the absolute charge-to-mass ratio for ion references.
    This is the paraxial reference-speed kick, with fixed x/y, z, dp and t0.
    A stationary cloud has no co-moving beam self-field cancellation factor.
    """
    xp, p = fields.xp, beam.particles
    if not isinstance(p.x, xp.ndarray):
        raise TypeError("ElectronCloud backend does not match the beam particle arrays")
    if not math.isfinite(interaction_length) or interaction_length < 0:
        raise ValueError("ElectronCloud interaction length must be finite and nonnegative")
    diagnostics = dict(n_alive=0, max_electric_field=0.0, max_kick=0.0, bunches=[])
    if interaction_length == 0 or fields.electron_density == 0:
        diagnostics["n_alive"] = sum(int(xp.count_nonzero(p.tag[b.start_idx:b.end_idx] > 0)) for b in beam.bunches)
        return diagnostics, []
    staged = []
    for bunch in beam.bunches:
        start, end = bunch.start_idx, bunch.end_idx
        indices = xp.flatnonzero(p.tag[start:end] > 0) + start
        n_alive = int(indices.size)
        if n_alive == 0:
            continue
        update, summary = _stage_cloud_slice(p, bunch, indices, fields.sample, interaction_length, xp)
        max_field, max_kick = summary["max_electric_field"], summary["max_kick"]
        diagnostics["n_alive"] += n_alive
        diagnostics["max_electric_field"] = max(diagnostics["max_electric_field"], max_field)
        diagnostics["max_kick"] = max(diagnostics["max_kick"], max_kick)
        diagnostics["bunches"].append(dict(bunch_id=int(bunch.bunch_id), n_alive=n_alive, max_electric_field=max_field, max_kick=max_kick))
        staged.append(update)
    return diagnostics, staged


def _stage_cloud_slice(p, bunch, indices, sample, interaction_length, xp):
    """Validate a sampled slice kick without changing particle coordinates."""
    beta, brho, charge = float(bunch.beta), float(bunch.brho), float(bunch.num_charge)
    if not (math.isfinite(beta) and 0 < beta <= 1 and math.isfinite(brho) and brho > 0 and math.isfinite(charge) and charge != 0):
        raise ValueError("ElectronCloud requires a finite positive reference speed/rigidity and a nonzero signed charge")
    x, y = p.x[indices], p.y[indices]
    px, py = p.px[indices], p.py[indices]
    momentum_ratio = 1.0 + p.dp[indices].astype(xp.float64)
    if not bool(xp.all(xp.isfinite(x) & xp.isfinite(y) & xp.isfinite(px) & xp.isfinite(py) & xp.isfinite(momentum_ratio))):
        raise ValueError("ElectronCloud live transverse particle coordinates must be finite")
    if not bool(xp.all((momentum_ratio > 0) & (px.astype(xp.float64)**2 + py.astype(xp.float64)**2 < momentum_ratio**2))):
        raise ValueError("ElectronCloud requires forward longitudinal particle momenta")
    ex, ey = sample(x, y)
    kick_factor = math.copysign(1.0, charge) * interaction_length / (beta * const.c * brho)
    kick_x, kick_y = kick_factor * ex, kick_factor * ey
    # Cast the staged result, not the kick, so a float32 update rounds once.
    next_px = xp.asarray(px.astype(xp.float64) + kick_x, dtype=p.px.dtype)
    next_py = xp.asarray(py.astype(xp.float64) + kick_y, dtype=p.py.dtype)
    if not bool(xp.all(xp.isfinite(next_px) & xp.isfinite(next_py))):
        raise ValueError("ElectronCloud kick produces non-finite transverse momenta")
    if not bool(xp.all(next_px.astype(xp.float64)**2 + next_py.astype(xp.float64)**2 < momentum_ratio**2)):
        raise ValueError("ElectronCloud kick exceeds the transverse thin-lens approximation; reduce the interaction length")
    max_field = float(xp.max(xp.hypot(ex, ey)))
    max_kick = float(xp.max(xp.hypot(kick_x, kick_y)))
    if not (math.isfinite(max_field) and math.isfinite(max_kick)):
        raise ValueError("ElectronCloud field or kick is non-finite")
    return (indices, next_px, next_py), dict(bunch_id=int(bunch.bunch_id), n_alive=int(indices.size), max_electric_field=max_field, max_kick=max_kick)


def apply_cloud_kicks(p, staged):
    """Commit previously validated transverse updates after required output."""
    for indices, next_px, next_py in staged:
        p.px[indices], p.py[indices] = next_px, next_py
