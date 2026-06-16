from PASS.utils.constants import const
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.helper import convert_keys_to_lower

import logging
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class BunchInfo:

    def __init__(self, input_file: str, bunch_id: int):

        self.start: int = 0
        self.stop: int = 0
        self._load_input(input_file, bunch_id)

    def _load_input(self, input_file: str, bunch_id: int) -> None:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = convert_keys_to_lower(data)
            
        bunch_data = data["sequence"]["injection"][f"bunch{bunch_id}"]

        self.bunch_id = bunch_id
        self.Ek = bunch_data["kinetic energy per nucleon (ev/u)"]
        self.Nrp = int(bunch_data["number of real particles"])
        self.Np = int(bunch_data["number of macro particles"])
        self.Np_sur = self.Np
        self.gamma_t = data.get("transition gamma")
        self.num_proton = int(data.get("number of protons"))
        self.num_neutron = int(data.get("number of neutrons"))
        self.num_charge = int(data.get("number of charges"))
        self.circum = data.get("circumference (m)")

        if self.Np == 0:
            raise ValueError(f"number of macro particles is zero for bunch {bunch_id}")
        if self.Nrp == 0:
            raise ValueError(f"number of real particles is zero for bunch {bunch_id}")
        self.ratio = self.Nrp / self.Np

        if self.num_proton == 0 and self.num_neutron == 0:  # electron or position
            if self.num_charge == -1:
                self.particle_type = "Electron"
            elif np.abs(self.num_charge) == 1:
                self.particle_type = "Position"
            else:
                raise ValueError(f"Incorrect charge number for electron or position: {self.num_charge}")
            self.m0 = const.m_e_eV
            self.qm_ratio = 1.0
        elif self.num_proton == 1 and self.num_neutron == 0:  # proton
            self.particle_type = "Proton"
            self.m0 = const.m_p_eV
            self.qm_ratio = 1.0
        else:  # other atomic nucleus
            self.particle_type = "Ion"
            self.m0 = const.m_u_eV
            self.qm_ratio = (np.abs(self.num_charge) / (self.num_proton + self.num_neutron))

        self.gamma = self.Ek / self.m0 + 1.0
        self.beta = np.sqrt(1.0 - 1.0 / self.gamma / self.gamma)
        # m0/c/c is in unit of eV, so gamma*m0/c/c*beta*c [unit: eV] = gamma*m0*beta/c [unit: eV] = gamma*m0*beta [unit:eV/c], so no need to multiply c again
        self.p0 = self.gamma * self.m0 * self.beta
        self.p0_kg = self.gamma * (self.m0 * const.e / (const.c * const.c)) * self.beta * const.c

        self.brho = self.p0_kg / (self.qm_ratio * const.e)

    def print(self) -> None:

        set_simple_logging()

        logger.info("")
        logger.info(center_string(s=f" Bunch{self.bunch_id} "))

        A = (self.num_proton or 0) + (self.num_neutron or 0)
        logger.info(f"Bunch ID: {self.bunch_id}")
        if self.particle_type == "Ion":
            logger.info(f"Particle Type: {self.particle_type} (Z={self.num_charge}), A={A}")
        else:
            logger.info(f"Particle Type: {self.particle_type}")
        logger.info(f"Kinetic Energy per Nucleon (MeV/u): {self.Ek/1e6:.6f}")
        logger.info(f"Number of Real Particles (1e9): {self.Nrp/1e9:.3f}")
        logger.info(f"Number of Macro Particles (1e6): {self.Np/1e6:.3f}")
        logger.info(f"Macro-to-Real ratio (1e3): {self.ratio/1e3:.6f}")
        logger.info(f"Rest mass per nucleon (MeV/c^2): {self.m0/1e6:.6f}")
        logger.info(f"Transition gamma: {self.gamma_t:.6f}")
        logger.info(f"Relativistic gamma: {self.gamma:.9f}")
        logger.info(f"Relativistic beta: {self.beta:.9f}")
        logger.info(f"Momentum per nucleon: {self.p0/1e6:.6f} MeV/c/u  ({self.p0_kg:.6e} kg·m/s/u)")
        logger.info(f"Circumference (m): {self.circum}")
        logger.info(f"BRho (T·m): {self.brho:.6f}")

        set_normal_logging()
