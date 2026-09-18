import logging

import numpy as np

from PASS.core.config import Config
from PASS.utils.program import LinearProgram
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.utils.constants import const
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string

logger = logging.getLogger(__name__)


class Beam:

    def __init__(self, input_file: str, cfg: Config):

        self.cfg = cfg
        self.use_gpu = cfg.use_gpu
        self.beam_id = cfg.input_path.index(input_file)
        self.beam_name = cfg.beam_name[self.beam_id]
        self.harmonic_number = cfg.harmonic_number[self.beam_id]
        self.bunches = []

        self._load_input()
        self._create_bunch_info()
        initialize_reference_clock(self, self._data)
        self._create_particles()

    def _load_input(self) -> None:
        data = self.cfg.input_data[self.beam_id]
        self.is_beambeam = data.get("is beam-beam", False)
        self.is_spaceCharge = bool(self.cfg.space_charge[self.beam_id].enabled)
        self._data = data

    def _create_bunch_info(self) -> None:
        inj = self._data["sequence"]["injection"]
        harmonic_ids = []
        for i in range(self.harmonic_number):
            key = f"bunch{i}"
            if key not in inj:
                raise ValueError(f"bunch{i} not declared in the injection configuration; "
                                 f"harmonic number {self.harmonic_number} requires "
                                 f"{self.harmonic_number} bunch dicts (one per group). "
                                 f"Declare empty bunches with 0 particles if a group is "
                                 f"unfilled.")
            bunch = BunchInfo(self._data, i)
            self.bunches.append(bunch)
            harmonic_ids.append(bunch.harmonic_id)

        expected_ids = set(range(self.harmonic_number))
        actual_ids = set(harmonic_ids)
        if len(actual_ids) != len(harmonic_ids) or actual_ids != expected_ids:
            raise ValueError("The injection bunch harmonic ids must be unique and cover "
                             f"[0, {self.harmonic_number}); got {harmonic_ids}. "
                             "The beam harmonic number is the bunch grouping count, so "
                             "declare one bunch dict per group slot.")

        self.Np_total = 0
        for bunch in self.bunches:
            bunch.start_idx = self.Np_total
            bunch.end_idx = self.Np_total + bunch.Np
            self.Np_total += bunch.Np

    def invalidate_slice_sets(self) -> None:
        """Invalidate all bunch-local slice results after regrouping."""
        for bunch in self.bunches:
            for slice_set in getattr(bunch, "slice_sets", {}).values():
                slice_set.invalidate()

    def _create_particles(self):
        if self.use_gpu:
            try:
                import cupy as cp
            except (ImportError, OSError) as exc:
                raise RuntimeError("The GPU backend was requested, but CuPy is unavailable. "
                                   "Install PASS with the optional [cuda] extra or select "
                                   "'cpu' in the input configuration.") from exc
            xp = cp
        else:
            xp = np

        dtype = np.float32 if self.cfg.particle_precision == "float32" else np.float64
        self.particles = ParticlePool(
            self.Np_total,
            xp,
            dtype=dtype,
        )
        # Beam storage reserves all planned particles; Injection activates each
        # batch by identity. ParticlePool itself remains useful for live scratch data.
        self.particles.tag.fill(0)

    def print(self) -> None:

        set_simple_logging()

        logger.info("")
        logger.info(center_string(s=f" Beam{self.beam_id} "))

        logger.info(f"Beam ID: {self.beam_id}")
        logger.info(f"Beam Name: {self.beam_name}")
        logger.info(f"Number of Bunches: {len(self.bunches)}")
        logger.info(f"Number of Total Macro Particles (1e6): {self.Np_total/1e6}")
        logger.info(f"Is Beam-Beam: {self.is_beambeam}")
        logger.info(f"Is Space-Charge: {self.is_spaceCharge}")

        set_normal_logging()

        for bunch in self.bunches:
            bunch.print()


def initialize_reference_clock(beam, data):
    """One prescribed grouping/RF clock; never follows a tracked bunch's energy."""
    values = {k.lower(): v for k, v in (data.get('reference clock') or {}).items()}
    initial = min(beam.bunches, key=lambda b: b.harmonic_id)
    frequency = values.get('revolution frequency (hz)', initial.beta * const.c / initial.circum)
    beam.reference_program = LinearProgram(frequency, values.get('time (s)'), origin=values.get('time origin (s)', 0.))
    if (np.any(beam.reference_program.values <= 0) or np.any(beam.reference_program.values * initial.circum >= const.c)):
        raise ValueError("Reference clock must define a positive subluminal design velocity")
    for b in beam.bunches:
        item = data['sequence']['injection'][f'bunch{b.bunch_id}']
        supplied = item.get('reference arrival time (s)')
        b.t0 = (float(supplied) if supplied is not None else beam.reference_program.inverse_integral(-b.harmonic_id / b.harmonic_number))
        if not np.isfinite(b.t0):
            raise ValueError("Reference arrival time must be finite")
