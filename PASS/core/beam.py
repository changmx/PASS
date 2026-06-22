from PASS.core.config import Config
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.helper import convert_keys_to_lower

import logging
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class Beam:

    def __init__(self, input_file: str, cfg: Config, is_cal_phase: bool = True):

        self.use_gpu = cfg.use_gpu
        self.beam_id = cfg.input_path.index(input_file)
        self.beam_name = cfg.beam_name[self.beam_id]
        self.num_bunch = cfg.num_bunch[self.beam_id]
        self.bunches = []
        self.is_cal_phase = is_cal_phase

        self._load_input(input_file)
        self._create_bunch_info(input_file)
        self._create_particles()

    def _load_input(self, input_file: str) -> None:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = convert_keys_to_lower(data)

        self.is_beambeam = data.get("is beam-beam", False)
        self.is_spaceCharge = data.get("is space charge", False)

    def _create_bunch_info(self, input_file: str) -> None:
        for i in range(self.num_bunch):
            self.bunches.append(BunchInfo(input_file, i))

        self.Np_total = 0
        for bunch in self.bunches:
            bunch.start_idx = self.Np_total
            bunch.end_idx = self.Np_total + bunch.Np
            self.Np_total += bunch.Np

    def _create_particles(self):
        if self.use_gpu:
            import cupy as cp
            xp = cp
        else:
            xp = np

        self.particles = ParticlePool(self.Np_total, xp, self.is_cal_phase)

    def print(self) -> None:

        set_simple_logging()

        logger.info("")
        logger.info(center_string(s=f" Beam{self.beam_id} "))

        logger.info(f"Beam ID: {self.beam_id}")
        logger.info(f"Beam Name: {self.beam_name}")
        logger.info(f"Number of Bunches: {self.num_bunch}")
        logger.info(f"Number of Total Macro Particles (1e6): {self.Np_total/1e6}")
        logger.info(f"Is Beam-Beam: {self.is_beambeam}")
        logger.info(f"Is Space-Charge: {self.is_spaceCharge}")

        set_normal_logging()

        for bunch in self.bunches:
            bunch.print()
