from PASS.core.config import Config
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Beam:

    def __init__(self, input_file: str, cfg: Config, is_cal_phase: bool = True):

        self.cfg = cfg
        self.use_gpu = cfg.use_gpu
        self.beam_id = cfg.input_path.index(input_file)
        self.beam_name = cfg.beam_name[self.beam_id]
        self.harmonic_number = cfg.harmonic_number[self.beam_id]
        self.bunches = []
        self.is_cal_phase = is_cal_phase

        self._load_input()
        self._create_bunch_info()
        self._create_particles()

    def _load_input(self) -> None:
        data = self.cfg.input_data[self.beam_id]
        self.is_beambeam = data.get("is beam-beam", False)
        self.is_spaceCharge = data.get("is space charge", False)
        self._data = data

    def _create_bunch_info(self) -> None:
        inj = self._data["sequence"]["injection"]
        harmonic_ids = []
        for i in range(self.harmonic_number):
            key = f"bunch{i}"
            if key not in inj:
                raise ValueError(
                    f"bunch{i} not declared in the injection configuration; "
                    f"harmonic number {self.harmonic_number} requires "
                    f"{self.harmonic_number} bunch dicts (one per group). "
                    f"Declare empty bunches with 0 particles if a group is "
                    f"unfilled."
                )
            bunch = BunchInfo(self._data, i)
            self.bunches.append(bunch)
            harmonic_ids.append(bunch.harmonic_id)

        expected_ids = set(range(self.harmonic_number))
        actual_ids = set(harmonic_ids)
        if len(actual_ids) != len(harmonic_ids) or actual_ids != expected_ids:
            raise ValueError(
                "The injection bunch harmonic ids must be unique and cover "
                f"[0, {self.harmonic_number}); got {harmonic_ids}. "
                "The beam harmonic number is the bunch grouping count, so "
                "declare one bunch dict per group slot."
            )

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
                raise RuntimeError(
                    "The GPU backend was requested, but CuPy is unavailable. "
                    "Install PASS with the optional [cuda] extra or select "
                    "'cpu' in the input configuration."
                ) from exc
            xp = cp
        else:
            xp = np

        dtype = np.float32 if self.cfg.particle_precision == "float32" else np.float64
        self.particles = ParticlePool(
            self.Np_total,
            xp,
            dtype=dtype,
            is_cal_phase=self.is_cal_phase,
        )

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
