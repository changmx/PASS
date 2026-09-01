"""SortBunch: sort all particles by longitudinal position and regroup them
into contiguous per-bunch index ranges.

Coordinate convention (per-bunch relative z):
    p.z stores the coordinate RELATIVE to the bunch's ideal particle
    (z_rel), and each BunchInfo carries z_center = harmonic_id * C / h_group
    (h_group = beam harmonic number / bunch-grouping multiplicity).  The
    laboratory position is z_lab = z_rel + z_center.

The azimuthal sort key is
    key = (z_lab + C/(2h_group)) mod C
which places bucket i particles in [i*C/h, (i+1)*C/h).  Sorting by key
therefore makes every bunch's particle indices contiguous, in ascending
bucket order, for any h (odd or even) and with no bunch split at a fold
boundary.  The bucket id of a sorted particle is floor(h*key/C).
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from PASS.commands.command import Command
from PASS.utils.logger import set_simple_logging, set_normal_logging

logger = logging.getLogger(__name__)

# Particle arrays that must be permuted together when sorting.
_ARRAY_NAMES = [
    "x", "px", "y", "py", "z", "dp", "tag",
    "lost_turn", "lost_position",
]
_OPTIONAL_ARRAY_NAMES = [
    "last_x", "last_px", "last_y", "last_py",
    "last_phasex", "last_phasey",
]


def _fold_by_ring(z, circumference):
    """Return the ring-period representative in [-C/2, C/2)."""
    return ((z + 0.5 * circumference) % circumference) - 0.5 * circumference


def bucket_sort_key(z_lab, h, circum):
    """Azimuthal sort key in [0, C) for the bucket grid of harmonic h."""
    half_bucket = 0.5 * circum / h
    return (z_lab + half_bucket) % circum


def bucket_id_from_key(key, h, circum):
    """Bucket id of a particle from its sort key: floor(h*key/C)."""
    return np.floor(h * key / circum).astype(np.int64)


def _permute_particle_arrays(beam, perm):
    """Reorder every particle array by the sorting permutation."""
    p = beam.particles
    for name in _ARRAY_NAMES + _OPTIONAL_ARRAY_NAMES:
        arr = getattr(p, name, None)
        if arr is not None:
            setattr(p, name, arr[perm])


def _invalidate_slice_sets(beam):
    """Invalidate bunch-local slice results after particle regrouping."""
    invalidate = getattr(beam, "invalidate_slice_sets", None)
    if invalidate is not None:
        invalidate()
        return
    for bunch in beam.bunches:
        for slice_set in getattr(bunch, "slice_sets", {}).values():
            slice_set.invalidate()


def _source_bunch_for_center(old_bunches, z_center: float, circum: float):
    """Choose the old reference whose center is nearest on the ring."""
    return min(
        old_bunches,
        key=lambda bunch: abs(_fold_by_ring(z_center - bunch.z_center, circum)),
    )


def _rebuild_bunches(beam, h_new: int):
    """Build one independently referenced BunchInfo object per new bucket."""
    old = beam.bunches
    circum = old[0].circum
    new_list = []
    for i in range(h_new):
        z_center = i * circum / h_new
        source = _source_bunch_for_center(old, z_center, circum)
        b = copy.deepcopy(source)
        b.bunch_id = i
        b.harmonic_number = h_new
        b.harmonic_id = i
        b.z_center = z_center
        b.start_idx = 0
        b.end_idx = 0
        b.Np = 0
        b.Nrp = 0
        new_list.append(b)
    beam.bunches = new_list


def regroup_particles(beam, new_harmonic: int | None = None):
    """Sort all particles by azimuth and make each bunch's indices contiguous.

    Without ``new_harmonic`` (SortBunch): bunches keep their harmonic ids and
    particles are grouped by their current bucket membership.

    With ``new_harmonic`` (ReorganizeBunch harmonic switch): the bucket grid
    is redefined to C/new_harmonic, the beam harmonic number is updated, the
    bunches are rebuilt (one per new bucket), and every particle is assigned
    to the nearest new bucket center.  Particles are renumbered contiguously
    per bunch.
    """
    p = beam.particles
    xp = p.xp
    C = beam.bunches[0].circum
    h_new = new_harmonic if new_harmonic is not None else beam.harmonic_number

    Np = beam.Np_total
    if Np == 0:
        if new_harmonic is not None:
            beam.harmonic_number = h_new
            _rebuild_bunches(beam, h_new)
        _invalidate_slice_sets(beam)
        return

    # --- laboratory longitudinal positions (old bunch centers) ---
    z_lab = xp.empty(Np, dtype=xp.float64)
    p0_by_particle = xp.empty(Np, dtype=xp.float64)
    weight_by_particle = xp.empty(Np, dtype=xp.float64)
    for b in beam.bunches:
        z_lab[b.start_idx:b.end_idx] = p.z[b.start_idx:b.end_idx] + b.z_center
        p0_by_particle[b.start_idx:b.end_idx] = b.p0
        weight_by_particle[b.start_idx:b.end_idx] = b.ratio

    # --- azimuthal sort key in [0, C) on the (possibly new) bucket grid ---
    key = bucket_sort_key(z_lab, h_new, C)
    perm = xp.argsort(key)
    key_sorted = key[perm]
    z_lab_sorted = z_lab[perm]
    p0_sorted = p0_by_particle[perm]
    weight_sorted = weight_by_particle[perm]

    _permute_particle_arrays(beam, perm)

    # --- rebuild bunch structure for a harmonic switch ---
    if new_harmonic is not None:
        beam.harmonic_number = h_new
        _rebuild_bunches(beam, h_new)

    # --- contiguous index ranges per bunch ---
    for b in beam.bunches:
        hid = b.harmonic_id
        start = int(xp.searchsorted(key_sorted, hid * C / h_new, side="left"))
        end = int(xp.searchsorted(key_sorted, (hid + 1) * C / h_new, side="left"))
        b.start_idx = start
        b.end_idx = end
        b.Np = end - start
        if b.Np > 0:
            b.Nrp = int(round(float(xp.sum(weight_sorted[start:end]))))
            b.ratio = b.Nrp / b.Np

        # --- recompute bunch-relative z against the assigned bucket center ---
        # Keep the nearest ring-period image.  This preserves bucket crossing
        # information at the grouping level without forcing particles back
        # into a bucket-width interval.
        p.z[start:end] = _fold_by_ring(z_lab_sorted[start:end] - b.z_center, C)

        if new_harmonic is not None and b.Np > 0:
            # px, py and dp are normalized to the old bunch reference.
            # Rebase them to the new bucket reference while preserving each
            # particle's absolute mechanical momentum.
            ref_scale = p0_sorted[start:end] / b.p0
            p.px[start:end] *= ref_scale
            p.py[start:end] *= ref_scale
            p.dp[start:end] = (
                (1.0 + p.dp[start:end]) * ref_scale - 1.0
            )

    _invalidate_slice_sets(beam)


@Command.register("sortbunch")
class SortBunch(Command):
    """Sort all particles by longitudinal position and regroup bunches.

    Reads the beam harmonic number (bunch grouping count), computes the
    azimuthal sort key, reorders every particle array, and assigns each
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
        regroup_particles(beam)
        return True

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, "
                    f"Name={self.cmd_name:s}")
        set_normal_logging()
