"""Regroup by machine-clock phase without folding the stored time coordinate.

At location s the key is (-Psi(t_i) + s/C + 1/(2h)) mod 1, where
Psi(t) is the integral of the prescribed revolution frequency and
t_i = bunch.t0 - z_i/(bunch.beta*c). Group j owns [j/h, (j+1)/h).
Only the key is periodic; live arrival times and mechanical momenta are
preserved when expressing particles in their destination bunch reference.
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from PASS.commands.command import Command
from PASS.core.bunch import set_reference_energy
from PASS.utils.constants import const
from PASS.utils.logger import set_simple_logging, set_normal_logging

logger = logging.getLogger(__name__)


def _permute_particle_arrays(beam, perm):
    """Reorder every particle array by the sorting permutation."""
    # Particle arrays that must be permuted together when sorting.
    array_names = [
        "x",
        "px",
        "y",
        "py",
        "z",
        "dp",
        "tag",
        "lost_turn",
        "lost_position",
    ]

    p = beam.particles
    for name in array_names:
        particle_array = getattr(p, name, None)
        if particle_array is not None:
            setattr(p, name, particle_array[perm])
    injection = getattr(beam, "injection_state", None)
    if injection is not None:
        injection.reorder(perm)


def _invalidate_slice_sets(beam):
    """Invalidate bunch-local slice results after particle regrouping."""
    invalidate = getattr(beam, "invalidate_slice_sets", None)
    if invalidate is not None:
        invalidate()
        return
    for bunch in beam.bunches:
        for slice_set in getattr(bunch, "slice_sets", {}).values():
            slice_set.invalidate()


def regroup_particles(beam, new_harmonic: int | None = None, *, location=0.):
    """Sort into complete, disjoint groups while preserving physical records.

    SortBunch retains the existing references. With ``new_harmonic``, choose
    new reference events and energies from the prescribed machine clock.
    This reference change leaves live particle time and momentum unchanged;
    lost and pending coordinates remain frozen. The common macro weight is
    fixed, and old bunch-local slice results are invalidated after regrouping.
    """
    p, xp = beam.particles, beam.particles.xp
    old_bunches = list(beam.bunches)
    n_particles = len(p.z)
    if (len(old_bunches) != beam.harmonic_number or not old_bunches
            or sorted(b.harmonic_id for b in old_bunches) != list(range(beam.harmonic_number))):
        raise ValueError("Bunch harmonic IDs must cover every grouping slot exactly once")
    # Check the source partition before filling temporary per-particle arrays.
    # Sort empty intervals before occupied intervals sharing the same start.
    expected_start = 0
    for b in sorted(old_bunches, key=lambda b: (b.start_idx, b.end_idx)):
        if not expected_start == b.start_idx <= b.end_idx <= n_particles:
            raise ValueError("Bunch ranges must cover all particles without gaps or overlaps")
        expected_start = b.end_idx
    if expected_start != n_particles:
        raise ValueError("Bunch ranges must cover all particles without gaps or overlaps")

    program = beam.reference_program
    C = old_bunches[0].circum
    h = beam.harmonic_number if new_harmonic is None else int(new_harmonic)
    if h < 1:
        raise ValueError("Grouping harmonic must be positive")
    epoch = min(b.t0 for b in old_bunches)
    arrival_time_from_epoch = xp.empty(n_particles, dtype=xp.float64)
    source_p0 = xp.empty(n_particles, dtype=xp.float64)
    keys = xp.empty(n_particles, dtype=xp.float64)
    for b in old_bunches:
        bunch_slice = slice(b.start_idx, b.end_idx)
        dt = -p.z[bunch_slice].astype(xp.float64) / (b.beta * const.c)
        # t_i - epoch stays unwrapped; the phase integral alone is reduced.
        arrival_time_from_epoch[bunch_slice] = (b.t0 - epoch) + dt
        keys[bunch_slice] = xp.remainder(-program.phase_cycles(b.t0, dt, xp) + location / C + .5 / h, 1.)
        source_p0[bunch_slice] = b.p0
    # A tiny negative argument can yield exactly 1.0 after floating remainder.
    # Keep that upper-side limit in the last group, rather than wrapping to 0.
    xp.minimum(keys, np.nextafter(1., 0.), out=keys)
    perm = xp.argsort(keys)
    keys = keys[perm]
    arrival_time_from_epoch = arrival_time_from_epoch[perm]
    source_p0 = source_p0[perm]

    # Adjacent groups share the same boundary, so ranges cannot overlap or
    # leave internal gaps. Transfer only h+1 indices from a GPU, once.
    boundaries = xp.searchsorted(keys, xp.arange(h + 1, dtype=xp.float64) / h, side="left")
    if xp is not np:
        boundaries = boundaries.get()
    if boundaries[0] != 0 or boundaries[-1] != n_particles:
        raise ValueError("Grouping keys must be finite and cover all particles in [0, 1)")

    destination_bunches = old_bunches
    if new_harmonic is not None:
        anchor = min(old_bunches, key=lambda b: b.harmonic_id)
        # Choose the nearest machine passage to the old anchor, then solve
        # Psi(T_j') = passage + s/C - j/h for the new reference events.
        passage = round(float(program.integral(anchor.t0)) - location / C + anchor.harmonic_id / anchor.harmonic_number)
        destination_bunches = []
        for hid in range(h):
            b = copy.deepcopy(anchor)
            b.bunch_id, b.harmonic_id, b.harmonic_number = hid, hid, h
            b.t0 = program.inverse_integral(passage + location / C - hid / h)
            beta = C * float(program.value(b.t0)) / const.c
            if not 0 < beta < 1:
                raise ValueError("Grouping clock must define a subluminal reference trajectory")
            set_reference_energy(b, b.m0 / np.sqrt(1 - beta * beta))
            destination_bunches.append(b)

    # Validate the partition and all new references before mutating beam state.
    _permute_particle_arrays(beam, perm)
    if new_harmonic is not None:
        beam.bunches = destination_bunches
        beam.harmonic_number = h
    for b in destination_bunches:
        start, end = map(int, boundaries[b.harmonic_id:b.harmonic_id + 2])
        b.start_idx, b.end_idx, b.Np = start, end, end - start
        bunch_slice = slice(start, end)
        # Np includes live, lost and pending records; Nrp is diagnostic only.
        # The common macro-particle weight is unchanged since initialization.
        b.Nrp = int(round(b.ratio * b.Np))
        live = p.tag[bunch_slice] > 0
        # z' = beta'*c*(T' - t_i), with no periodic reduction of stored z.
        p.z[bunch_slice] = xp.where(live, b.beta * const.c * ((b.t0 - epoch) - arrival_time_from_epoch[bunch_slice]), p.z[bunch_slice])
        scale = source_p0[bunch_slice] / b.p0
        p.px[bunch_slice] = xp.where(live, p.px[bunch_slice].astype(xp.float64) * scale, p.px[bunch_slice])
        p.py[bunch_slice] = xp.where(live, p.py[bunch_slice].astype(xp.float64) * scale, p.py[bunch_slice])
        # P0'*(1+dp') = P0*(1+dp); avoid losing tiny dp by first adding 1.
        p.dp[bunch_slice] = xp.where(live, p.dp[bunch_slice].astype(xp.float64) * scale + (source_p0[bunch_slice] - b.p0) / b.p0, p.dp[bunch_slice])
    # Old per-bunch indices no longer describe these populations. No rebinning
    # is performed; users must explicitly generate slices for the new grouping.
    _invalidate_slice_sets(beam)


@Command.register("sortbunch")
class SortBunch(Command):
    """Sort all particles by machine-clock phase and regroup bunches.

    Reads the beam harmonic number (bunch grouping count), computes the
    periodic phase key, reorders every particle array, and assigns each
    bunch a contiguous index range (start_idx/end_idx/Np).
    """

    def __init__(self, beam_id: int, sim, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}
        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = 0.0
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]
        super().__init__()

    def execute_cpu(self, sim):
        return self._sort(sim)

    def execute_gpu(self, sim):
        return self._sort(sim)

    def _sort(self, sim):
        beam = sim.beams[self.beam_id]
        set_simple_logging()
        logger.info(f"[SortBunch] {self.cmd_name}: sorting {beam.Np_total} "
                    f"particles, h={beam.harmonic_number}")
        set_normal_logging()
        regroup_particles(beam, location=self.s)
        return True

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, "
                    f"Name={self.cmd_name:s}")
        set_normal_logging()
