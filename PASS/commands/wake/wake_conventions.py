"""Canonical wake quantities; transport coordinates are never folded."""
import numpy as np

from PASS.utils.constants import const


def signed_charge_per_mass_unit(bunch):
    return bunch.num_charge / max(bunch.num_proton + bunch.num_neutron, 1)


def arrival_times(z_rel, bunch):
    if not 0 < bunch.beta <= 1:
        raise ValueError("WakeField requires 0 < reference beta <= 1")
    return bunch.t0 - (np.asarray(z_rel, dtype=float) + bunch.z_center) / (bunch.beta * const.c)


def require_cpu(backend):
    if backend == "gpu":
        raise TypeError("This CPU entry point requires host arrays; use its GPU counterpart")
    if backend != "cpu":
        raise ValueError(f"Unknown wake backend {backend!r}")


def apply_kick_cpu(particles, bunch, indices, voltage, *, s, turn):
    """V=(positive longitudinal loss, +x force, +y force), in volts.

    PASS p0 and energy are per nucleon (per particle for electrons).
    The witness macro weight cancels against its momentum/energy weight.
    """
    z_over_a = signed_charge_per_mass_unit(bunch)
    p0 = bunch.p0
    old_p = (1 + np.asarray(particles.dp[indices], dtype=float)) * p0
    old_energy = np.hypot(old_p, bunch.m0)
    energy_change = -z_over_a * voltage[0]
    energy = old_energy + energy_change
    momentum_squared = (energy - bunch.m0) * (energy + bunch.m0)
    # Convert the integrated transverse Lorentz-force voltage to mechanical
    # impulse using the incoming witness speed, including momentum spread.
    beta_witness = old_p/old_energy
    inverse = np.divide(z_over_a, beta_witness*p0, out=np.zeros_like(old_p), where=beta_witness > 0)
    px = particles.px[indices] + voltage[1]*inverse
    py = particles.py[indices] + voltage[2]*inverse
    momentum = np.sqrt(np.maximum(momentum_squared, 0))
    # Rationalize p_new - p_old to retain weak longitudinal kicks. Computing
    # p_new/p0 - 1 directly discards changes much smaller than the reference
    # momentum and introduces round-off even with zero longitudinal voltage.
    denominator = p0*(momentum+old_p)
    increment = np.divide(energy_change*(2*old_energy+energy_change), denominator,
                          out=np.zeros_like(denominator), where=denominator != 0)
    dp = particles.dp[indices]+increment
    # A purely transverse wake conserves momentum magnitude in these
    # coordinates. Avoid accumulating round-trip energy/momentum rounding.
    dp = np.where(voltage[0] == 0, particles.dp[indices], dp)
    valid = ((old_p > 0) & (energy > bunch.m0) & np.isfinite(dp) & np.isfinite(px)
             & np.isfinite(py) & ((1 + dp)**2 > px**2 + py**2))
    good, bad = indices[valid], indices[~valid]
    particles.dp[good] = dp[valid]
    particles.px[good] = px[valid]
    particles.py[good] = py[valid]
    particles.tag[bad] = -np.abs(particles.tag[bad])
    particles.lost_position[bad] = s
    particles.lost_turn[bad] = turn


def apply_kick_gpu(particles, bunch, indices, voltage, *, s, turn):
    """Array-level API; WakeField uses its faster fused monomial/kick kernel."""
    import cupy as cp
    indices = cp.asarray(indices)
    voltage = cp.asarray(voltage, dtype=cp.float64)
    za, p0 = signed_charge_per_mass_unit(bunch), bunch.p0
    oldp = (1+particles.dp[indices].astype(cp.float64))*p0
    olde = cp.hypot(oldp, bunch.m0)
    de = -za*voltage[0]
    energy = olde+de
    momentum = cp.sqrt(cp.maximum((energy-bunch.m0)*(energy+bunch.m0), 0.))
    beta = oldp/olde
    inverse = cp.where(beta > 0, za/(cp.where(beta > 0, beta, 1.)*p0), 0.)
    px = particles.px[indices]+voltage[1]*inverse
    py = particles.py[indices]+voltage[2]*inverse
    denominator = p0*(momentum+oldp)
    delta = particles.dp[indices]+cp.where((voltage[0] != 0) & (denominator != 0),
        de*(2*olde+de)/cp.where(denominator != 0, denominator, 1.), 0.)
    valid = ((oldp > 0) & (energy > bunch.m0) & cp.isfinite(delta) & cp.isfinite(px)
        & cp.isfinite(py) & ((1+delta)**2 > px*px+py*py))
    good, bad = indices[valid], indices[~valid]
    particles.dp[good], particles.px[good], particles.py[good] = delta[valid], px[valid], py[valid]
    particles.tag[bad] = -cp.abs(particles.tag[bad])
    particles.lost_position[bad], particles.lost_turn[bad] = s, turn
