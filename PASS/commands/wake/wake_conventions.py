"""Canonical wake quantities; transport coordinates are never folded."""
import numpy as np

from PASS.utils.constants import const


def signed_charge_per_mass_unit(bunch):
    return bunch.num_charge / max(bunch.num_proton + bunch.num_neutron, 1)


def arrival_times(z_rel, bunch):
    if not 0 < bunch.beta <= 1:
        raise ValueError("WakeField requires 0 < reference beta <= 1")
    return bunch.t0 - np.asarray(z_rel, dtype=float) / (bunch.beta * const.c)


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
    beta_witness = old_p / old_energy
    inverse = np.divide(z_over_a, beta_witness * p0, out=np.zeros_like(old_p), where=beta_witness > 0)
    px = particles.px[indices] + voltage[1] * inverse
    py = particles.py[indices] + voltage[2] * inverse
    momentum = np.sqrt(np.maximum(momentum_squared, 0))
    # Rationalize p_new - p_old to retain weak longitudinal kicks. Computing
    # p_new/p0 - 1 directly discards changes much smaller than the reference
    # momentum and introduces round-off even with zero longitudinal voltage.
    denominator = p0 * (momentum + old_p)
    increment = np.divide(energy_change * (2 * old_energy + energy_change), denominator, out=np.zeros_like(denominator), where=denominator != 0)
    dp = particles.dp[indices] + increment
    # A purely transverse wake conserves momentum magnitude in these
    # coordinates. Avoid accumulating round-trip energy/momentum rounding.
    dp = np.where(voltage[0] == 0, particles.dp[indices], dp)
    valid = ((old_p > 0) & (energy > bunch.m0) & np.isfinite(dp) & np.isfinite(px) & np.isfinite(py) & ((1 + dp)**2 > px**2 + py**2))
    good, bad = indices[valid], indices[~valid]
    particles.dp[good] = dp[valid]
    particles.px[good] = px[valid]
    particles.py[good] = py[valid]
    particles.tag[bad] = -np.abs(particles.tag[bad])
    particles.lost_position[bad] = s
    particles.lost_turn[bad] = turn


def apply_kick_gpu(particles, bunch, indices, voltage, *, s, turn):
    """Sparse array API sharing the command's mechanical CUDA kick formula."""
    import cupy as cp
    from PASS.commands.wake_field import _GPU_CODE
    indices = cp.ascontiguousarray(cp.asarray(indices), dtype=cp.int64)
    voltage = cp.ascontiguousarray(cp.asarray(voltage), dtype=cp.float64)
    n = len(indices)
    if voltage.shape != (3, n):
        raise ValueError('Wake voltages must have shape (3, number of witnesses)')
    cache = particles.__dict__.setdefault('_wake_sparse_kernels', {})
    key = (cp.cuda.runtime.getDevice(), np.dtype(particles.dtype))
    if key not in cache:
        cache[key] = cp.RawKernel(_GPU_CODE + _SPARSE_KICK,
                                  'sparse_kick',
                                  options=('--std=c++17', f'-DFLOAT_PARTICLES={int(particles.dtype==np.float32)}'))
    if n:
        cache[key](
            ((n + 255) // 256, ), (256, ),
            (particles.px, particles.py, particles.dp, particles.tag, particles.lost_turn, particles.lost_position, indices, voltage, np.int64(n),
             np.float64(bunch.p0), np.float64(bunch.m0), np.float64(signed_charge_per_mass_unit(bunch)), np.float64(s), np.int32(turn)))


_SPARSE_KICK = r'''
extern "C" __global__ void sparse_kick(
    R* px,
    R* py,
    R* dp,
    int* tag,
    int* lost_turn,
    float* lost_position,
    const long long* indices,
    const double* voltage,
    long long n,
    double p0,
    double m0,
    double za,
    double position,
    int turn
) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n)
        return;
    double v[3] = {voltage[j], voltage[n + j], voltage[2 * n + j]};
    mechanical(indices[j], px, py, dp, tag, lost_turn, lost_position, v, p0, m0, za, position, turn);
}
'''
