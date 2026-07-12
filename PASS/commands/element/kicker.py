from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.aperture import check_aperture_cpu

import numpy as np
import cupy as cp
import logging

logger = logging.getLogger(__name__)


@Command.register("kicker")
class Kicker(Command):
    """
    Kicker magnet (thin lens dipole kick).

    A kicker is a pulsed dipole magnet that applies a transverse angular kick
    to the beam. It is modelled as a thin lens (length=0): a pure momentum
    translation that is exactly symplectic.

    Kick formula (integrated dipole, order-0 multipole):

      Δpx = hkick
      Δpy = vkick

    where hkick and vkick are the integrated dipole strengths (in radians),
    equivalent to MAD-X's hkick/vkick and to Multipole with knl=[hkick],
    ksl=[vkick].

    Both horizontal and vertical kicks can be non-zero simultaneously
    (bipolar kicker), or only one of them (unidirectional kicker).

    Coordinate convention (PASS):
      x, px, y, py, z, dp(=δ)
      px = Px/P0,  py = Py/P0,  dp = (P-P0)/P0
      z  = s - β0·c·t  (ζ coordinate)
    """

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.length = 0.0
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        # Kick strengths (radians)
        self.hkick: float = kwargs.get("hkick", 0.0)
        self.vkick: float = kwargs.get("vkick", 0.0)

        if abs(self.hkick) < const.eps and abs(self.vkick) < const.eps:
            logger.warning(f"Kicker {self.cmd_name} has zero kick strength "
                           f"(hkick={self.hkick}, vkick={self.vkick}). It will act as a marker.")

        # --- aperture ---
        self.aperture_type: str = kwargs.get("aperture type", "off").lower()
        self.aperture_value: list = kwargs.get("aperture value", [])
        if not isinstance(self.aperture_value, list):
            raise ValueError(f"Aperture value of {self.cmd_name} must be a list, but got {type(self.aperture_value)}")

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, "
                    f"Hkick={self.hkick:.6e}, Vkick={self.vkick:.6e}, "
                    f"ApertureType={self.aperture_type:s}, ApertureValue={self.aperture_value}")
        set_normal_logging()

    def execute_cpu(self, sim):
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            self._kicker_kick_cpu(beam, bunch)
            check_aperture_cpu(beam, bunch, self.aperture_type, self.aperture_value, self.s, turn)

    def execute_gpu(self, sim):
        raise NotImplementedError("GPU implementation of Kicker is not yet available")

    # ------------------------------------------------------------------
    # Kicker kick (CPU)
    # ------------------------------------------------------------------

    def _kicker_kick_cpu(self, beam: Beam, bunch: BunchInfo):
        """Apply the thin-lens dipole kick to particles (CPU).

        Δpx = hkick
        Δpy = vkick

        Only alive particles (tag > 0) receive the kick.
        """
        start = bunch.start_idx
        end = bunch.end_idx

        p = beam.particles
        px = p.px[start:end]
        py = p.py[start:end]
        tag = p.tag[start:end]

        mask = (tag > 0).astype(np.float64)

        px += self.hkick * mask
        py += self.vkick * mask
