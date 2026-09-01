from PASS.utils.constants import const
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BunchInfo:

    def __init__(self, input_data: dict, bunch_id: int):

        self.start_idx: int = 0
        self.end_idx: int = 0
        self.slice_sets: dict[str, object] = {}
        self._load_input(input_data, bunch_id)

    def _load_input(self, data: dict, bunch_id: int) -> None:
        bunch_data = data["sequence"]["injection"][f"bunch{bunch_id}"]

        self.bunch_id = bunch_id
        self.Ek = bunch_data["kinetic energy per nucleon (ev/u)"]
        self.Nrp = int(bunch_data["number of real particles"])
        self.Np = int(bunch_data["number of macro particles"])
        self.sigma_z = bunch_data["sigma z (m)"]
        self.dp = bunch_data["sigma dp/p"]

        self.Np_sur = self.Np
        self.gamma_t = data.get("transition gamma")
        self.num_proton = int(data.get("number of protons"))
        self.num_neutron = int(data.get("number of neutrons"))
        self.num_charge = int(data.get("number of charges"))
        self.circum = data.get("circumference (m)")

        if self.Np == 0:
            self.ratio = 0.0
        else:
            self.ratio = self.Nrp / self.Np

        # --- Bunch grouping metadata (per-bunch relative z convention) ---
        # The beam harmonic number is declared once at the injection level
        # and means how many longitudinal groups the beam is organized into.
        self.harmonic_number = int(
            data["sequence"]["injection"]["harmonic number"]
        )
        self.harmonic_id = int(bunch_data.get("harmonic id of this bunch", 0))
        if not (0 <= self.harmonic_id < self.harmonic_number):
            raise ValueError(
                f"bunch {bunch_id}: harmonic id {self.harmonic_id} out of "
                f"range [0, {self.harmonic_number})"
            )
        # Ideal-particle (bunch-center) longitudinal position in the machine
        # reference frame.  Particle coordinates p.z are stored RELATIVE to
        # this position (z_rel), so z_lab = p.z + z_center.
        self.z_center = self.harmonic_id * self.circum / self.harmonic_number

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

        self.t0 = 0.0

        self._register_slice_sets(data)

    def _register_slice_sets(self, data: dict) -> None:
        """Register all Slice command configurations for this bunch.

        Every bunch gets the same named configuration.  The resulting
        SliceSet objects are independent so their runtime IDs and statistics
        cannot overwrite another bunch or another collective effect.
        """
        # Delayed import avoids the commands package importing BunchInfo while
        # the core beam/bunch modules are still being initialized.
        from PASS.commands.slicer import SliceSet

        sequence = data.get("sequence", {})
        for command_name, command_data in sequence.items():
            if not isinstance(command_data, dict):
                continue
            if str(command_data.get("command", "")).strip().lower() != "slice":
                continue

            name = command_data.get("slice set")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Slice command {command_name!r} requires a non-empty "
                    "'Slice set' name"
                )
            name = name.strip()
            candidate = SliceSet(
                name=name,
                model=command_data.get("slice model", "equal_length"),
                num_slices=command_data.get("number of slices", 100),
                z_range_mode=command_data.get("z range mode", "auto_sigma"),
                z_min=command_data.get("z min"),
                z_max=command_data.get("z max"),
                num_sigma=command_data.get("number of sigma", 6.0),
                source_command=command_name,
            )
            previous = self.slice_sets.get(name)
            if previous is not None:
                if previous.configuration() != candidate.configuration():
                    raise ValueError(
                        f"Slice set {name!r} is configured inconsistently by "
                        f"commands {previous.source_command!r} and "
                        f"{command_name!r}"
                    )
                continue
            self.slice_sets[name] = candidate

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
