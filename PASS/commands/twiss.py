from PASS.commands.command import Command
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.core.config import Config
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const

import numpy as np
import cupy as cp
import logging

logger = logging.getLogger(__name__)


@Command.register("twiss")
class Twiss(Command):

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id: int = beam_id
        self.s: float = kwargs["s (m)"]
        self.cmd_type: str = self.__class__.__name__
        self.cmd_name: str = kwargs["name"]

        self.s_previous: float = kwargs["s previous (m)"]

        self.alphax: float = kwargs["alpha x"]
        self.alphay: float = kwargs["alpha y"]
        self.alphax_previous: float = kwargs["alpha x previous"]
        self.alphay_previous: float = kwargs["alpha y previous"]

        self.betax: float = kwargs["beta x (m)"]
        self.betay: float = kwargs["beta y (m)"]
        self.betax_previous: float = kwargs["beta x previous (m)"]
        self.betay_previous: float = kwargs["beta y previous (m)"]

        self.mux: float = kwargs["mu x"]
        self.muy: float = kwargs["mu y"]
        self.mux_previous: float = kwargs["mu x previous"]
        self.muy_previous: float = kwargs["mu y previous"]

        self.Dx: float = kwargs["dx (m)"]
        self.Dx_previous: float = kwargs["dx previous (m)"]

        self.Dpx: float = kwargs["dpx"]
        self.Dpx_previous: float = kwargs["dpx previous"]

        self.DQx: float = kwargs["dqx"]
        self.DQy: float = kwargs["dqy"]

        self.longitudinal_transfer: str = kwargs["longitudinal transfer"].lower()

        self.muz: float = kwargs.get("mu z", 0.0)
        self.muz_previous: float = kwargs.get("mu z previous", 0.0)

        self.phi_x: float = (self.mux - self.mux_previous) * 2.0 * const.pi
        self.phi_y: float = (self.muy - self.muy_previous) * 2.0 * const.pi
        self.phi_z: float = (self.muz - self.muz_previous) * 2.0 * const.pi

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(
            f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, S_previous={self.s_previous:.4f}, "
            f"alphax={self.alphax:.4f}, alphay={self.alphay:.4f}, alphax_previous={self.alphax_previous:.4f}, alphay_previous={self.alphay_previous:.4f}, "
            f"betax={self.betax:.4f}, betay={self.betay:.4f}, betax_previous={self.betax_previous:.4f}, betay_previous={self.betay_previous:.4f}, "
            f"mux={self.mux:.4f}, muy={self.muy:.4f}, muz={self.muz:.4f}, "
            f"Dx={self.Dx:.4f}, Dpx={self.Dpx:.4f}, "
            f"DQx={self.DQx:.4f}, DQy={self.DQy:.4f}, "
            f"longitudinal_transfer={self.longitudinal_transfer:s}")
        set_normal_logging()

    def execute_cpu(self, sim):

        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        turn = sim.state.turn

        for i, bunch in enumerate(bunches):
            dt = (self.s - self.s_previous) / (bunch.beta * const.c)
            bunch.t0 += dt

            if self.longitudinal_transfer == "drift":
                gammat = bunch.gamma_t
                gamma = bunch.gamma
                m11_z = 1.0
                m12_z = -1.0 * (1.0 / gammat**2 - 1.0 / gamma**2) * (self.s - self.s_previous)
                m21_z = 0.0
                m22_z = 1.0
            elif self.longitudinal_transfer == "matrix":
                sigmaz = bunch.sigma_z
                dp_bunch = bunch.dp
                m11_z = np.cos(self.phi_z)
                m12_z = sigmaz / dp_bunch * np.sin(self.phi_z)
                m21_z = -dp_bunch / sigmaz * np.sin(self.phi_z)
                m22_z = np.cos(self.phi_z)
            else:
                m11_z = 1.0
                m12_z = 0.0
                m21_z = 0.0
                m22_z = 1.0

            circum = bunch.circum
            start = bunch.start_idx
            end = bunch.end_idx

            p = beam.particles
            x = p.x[start:end]
            px = p.px[start:end]
            y = p.y[start:end]
            py = p.py[start:end]
            z = p.z[start:end]
            dp = p.dp[start:end]
            tag = p.tag[start:end]
            lost_turn = p.lost_turn[start:end]
            lost_position = p.lost_position[start:end]

            alive = tag > 0

            z2 = z * m11_z + dp * m12_z
            dp2 = z * m21_z + dp * m22_z

            x1 = x - self.Dx_previous * dp
            px1 = px - self.Dpx_previous * dp

            y1 = y
            py1 = py

            phi_x = self.phi_x + dp * self.DQx * 2.0 * const.pi
            phi_y = self.phi_y + dp * self.DQy * 2.0 * const.pi

            cx = np.cos(phi_x)
            sx = np.sin(phi_x)
            cy = np.cos(phi_y)
            sy = np.sin(phi_y)

            sqrt_betax_betaxprev = np.sqrt(self.betax * self.betax_previous)
            sqrt_betax_de_betaxprev = np.sqrt(self.betax / self.betax_previous)
            sqrt_betaxprev_de_betax = np.sqrt(self.betax_previous / self.betax)

            sqrt_betay_betayprev = np.sqrt(self.betay * self.betay_previous)
            sqrt_betay_de_betayprev = np.sqrt(self.betay / self.betay_previous)
            sqrt_betayprev_de_betay = np.sqrt(self.betay_previous / self.betay)

            m11_x = sqrt_betax_de_betaxprev * (cx + self.alphax_previous * sx)
            m12_x = sqrt_betax_betaxprev * sx
            m21_x = -(1.0 + self.alphax * self.alphax_previous) / sqrt_betax_betaxprev * sx \
                + (self.alphax_previous - self.alphax) / sqrt_betax_betaxprev * cx
            m22_x = sqrt_betaxprev_de_betax * (cx - self.alphax * sx)

            m11_y = sqrt_betay_de_betayprev * (cy + self.alphay_previous * sy)
            m12_y = sqrt_betay_betayprev * sy
            m21_y = -(1.0 + self.alphay * self.alphay_previous) / sqrt_betay_betayprev * sy \
                + (self.alphay_previous - self.alphay) / sqrt_betay_betayprev * cy
            m22_y = sqrt_betayprev_de_betay * (cy - self.alphay * sy)

            x2 = x1 * m11_x + px1 * m12_x + self.Dx * dp2
            px2 = x1 * m21_x + px1 * m22_x + self.Dpx * dp2

            y2 = y1 * m11_y + py1 * m12_y
            py2 = y1 * m21_y + py1 * m22_y

            c_half = 0.5 * circum
            over = (z2 > c_half).astype(np.int64)
            under = (z2 < -c_half).astype(np.int64)
            z2 += (under - over) * circum

            # --- write back (only alive particles) ---
            z[:] = np.where(alive, z2, z)
            dp[:] = np.where(alive, dp2, dp)
            x[:] = np.where(alive, x2, x)
            px[:] = np.where(alive, px2, px)
            y[:] = np.where(alive, y2, y)
            py[:] = np.where(alive, py2, py)

            # --- mark lost particles ---
            lost = alive & ((np.abs(x2) > 1.0) | (np.abs(y2) > 1.0))
            if np.any(lost):
                tag[lost] = -np.abs(tag[lost])
                lost_turn[lost] = turn
                lost_position[lost] = self.s

    def execute_gpu(self, sim):
        pass
