from __future__ import annotations

from PASS.commands.command import Command
from PASS.core.config import Config
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.helper import get_current_time

import numpy as np
import cupy as cp
import pandas as pd
import logging
import tfs
import re
from pathlib import Path
import random
from scipy.optimize import brentq
from scipy.integrate import dblquad
import os

logger = logging.getLogger(__name__)


@Command.register("injection")
class Injection(Command):

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        if np.abs(self.s) > const.eps:
            raise ValueError(f"The position s of injection must be 0, but now is {self.s}")

        self.num_bunch = len(kwargs) - 2
        self.inj_bunchs = []
        for bunch_id in range(self.num_bunch):
            inj_bunch = InjectionBunchInfo(self.beam_id, bunch_id, sim, **kwargs[f"bunch{bunch_id}"])
            self.inj_bunchs.append(inj_bunch)

        self.rng = random.Random()

        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}")
        for inj_bunch in self.inj_bunchs:
            inj_bunch.print()
        set_normal_logging()

    def execute_cpu(self, sim: Simulation):
        self._execute(sim)

    def execute_gpu(self, sim: Simulation):
        self._execute(sim)

    def _execute(self, sim: Simulation):
        cfg = sim.cfg
        beam = sim.beams[self.beam_id]
        bunches: list[BunchInfo] = beam.bunches
        state = sim.state

        use_cpu = cfg.use_cpu
        turn = state.turn

        for i in range(beam.num_bunch):
            inj_bunch = self.inj_bunchs[i]
            if turn not in inj_bunch.inj_turns:
                continue
            bunch_info = bunches[i]
            Np = bunch_info.Np
            total_inj_turns = len(inj_bunch.inj_turns)
            inj_bunch.Np_inj_curTurn = int(Np / total_inj_turns)
            logger.info(f"total injection turns = {total_inj_turns}, Np_inj_curTurn = {inj_bunch.Np_inj_curTurn}")
            if (turn == inj_bunch.inj_turns[0] and inj_bunch.Np_inj_curTurn * total_inj_turns != Np):
                inj_bunch.Np_inj_curTurn += (Np - inj_bunch.Np_inj_curTurn * total_inj_turns)
                logger.info(
                    f"[Injection] Since the total number of particles {Np} cannot be divided exactly by the number of injection turns {total_inj_turns}, we will inject {inj_bunch.Np_inj_curTurn} particles in the first turn and {Np / total_inj_turns} particles in the rest turns."
                )

            if inj_bunch.is_load_dist:
                self._load_dist(inj_bunch, bunch_info, beam, use_cpu)
            else:
                if inj_bunch.dist_trans.lower() == "kv":
                    self._generate_trans_kv_dist(inj_bunch, bunch_info, beam, use_cpu)
                elif inj_bunch.dist_trans.lower() == "gaussian":
                    self._generate_trans_gaussian_dist(inj_bunch, bunch_info, beam, use_cpu)
                elif inj_bunch.dist_trans.lower() == "uniform":
                    self._generate_trans_uniform_dist(inj_bunch, bunch_info, beam, use_cpu)
                else:
                    raise ValueError(f"We don't support transverse distribution: {inj_bunch.dist_trans}")

                if inj_bunch.dist_longi.lower() == "gaussian":
                    self._generate_longi_gaussian_dist(inj_bunch, bunch_info, beam, use_cpu)
                elif inj_bunch.dist_longi.lower() == "coasting":
                    self._generate_longi_coasting_dist(inj_bunch, bunch_info, beam, use_cpu)
                elif inj_bunch.dist_longi.lower() == "matchz":
                    self._generate_longi_matchZ_dist(inj_bunch, bunch_info, beam, use_cpu)
                elif inj_bunch.dist_longi.lower() == "matchdp":
                    self._generate_longi_matchDp_dist(inj_bunch, bunch_info, beam, use_cpu)
                else:
                    raise ValueError(f"We don't support longitudinal distribution: {inj_bunch.dist_longi}")

            self._add_offset(inj_bunch, bunch_info, beam, use_cpu)

            inj_bunch.Np_injected += inj_bunch.Np_inj_curTurn

            if turn == inj_bunch.inj_turns[0]:
                if inj_bunch.is_insert_particles:
                    self._insert_particles(inj_bunch, bunch_info, beam, use_cpu)

            if turn == inj_bunch.inj_turns[-1]:
                if inj_bunch.is_save_init_dist:
                    self._save_init_dist(inj_bunch, bunch_info, beam, cfg)

    def _load_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):

        path = Path(inj_bunch.load_dist_filepath)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        logger.info(f"Start loading data from file: {path} ...")

        df = tfs.read(path)
        x = df["x"].to_numpy()
        px = df["px"].to_numpy()
        y = df["y"].to_numpy()
        py = df["py"].to_numpy()
        z = df["z"].to_numpy()
        pz = df["pz"].to_numpy()

        len_input = len(x)

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn

        copy_start = start_index
        copy_end = min(end_index, len_input)

        if copy_start >= len_input:
            logger.warning(f"No more particles to inject from file {path}. Start index {copy_start} beyond file length {len_input}")
        else:
            df_start = copy_start
            df_end = copy_end

            p = beam.particles
            p.x[copy_start:copy_end] = p.xp.asarray(x[df_start:df_end])
            p.px[copy_start:copy_end] = p.xp.asarray(px[df_start:df_end])
            p.y[copy_start:copy_end] = p.xp.asarray(y[df_start:df_end])
            p.py[copy_start:copy_end] = p.xp.asarray(py[df_start:df_end])
            p.z[copy_start:copy_end] = p.xp.asarray(z[df_start:df_end])
            p.pz[copy_start:copy_end] = p.xp.asarray(pz[df_start:df_end])

            if copy_end < end_index:
                logger.warning(f"Only copy particles {copy_start}-{copy_end} from file: {path}")

        logger.info("Successfully")

    def _generate_trans_kv_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):
        # This menthod is derived from "Particle - in - cell code BEAMPATH for beam dynamics simulations in linear accelerators and beamlines"
        # The two beams shoule have different seed values to generate different random values.
        # This is 4-D generator.

        logger.info(f"The initial transverse KV distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        Np_inj = inj_bunch.Np_inj_curTurn

        emit_x = inj_bunch.emitx
        emit_y = inj_bunch.emity
        alpha_x_twiss = inj_bunch.alphax
        alpha_y_twiss = inj_bunch.alphay
        beta_x_twiss = inj_bunch.betax
        beta_y_twiss = inj_bunch.betay
        gamma_x_twiss = inj_bunch.gammax
        gamma_y_twiss = inj_bunch.gammay

        sigma_x = inj_bunch.sigmax
        sigma_y = inj_bunch.sigmay

        # ε/ε_rms = 1,  [-1 sigma, 1 sigma], the x-px phase space contains: 39.346934029%
        # ε/ε_rms = 4,  [-2 sigma, 2 sigma], the x-px phase space contains: 86.466471676%
        # ε/ε_rms = 9,  [-3 sigma, 3 sigma], the x-px phase space contains: 98.889100346%
        # ε/ε_rms = 16, [-4 sigma, 4 sigma], the x-px phase space contains: 99.966453737%
        # ε/ε_rms = 25, [-5 sigma, 5 sigma], the x-px phase space contains: 99.999627335%
        # ε/ε_rms = 36, [-6 sigma, 6 sigma], the x-px phase space contains: 99.999998477%
        x_max = 4 * sigma_x
        x_min = -4 * sigma_x
        y_max = 4 * sigma_y
        y_min = -4 * sigma_y

        x_arr = np.zeros(Np_inj, dtype=np.float64)
        px_arr = np.zeros(Np_inj, dtype=np.float64)
        y_arr = np.zeros(Np_inj, dtype=np.float64)
        py_arr = np.zeros(Np_inj, dtype=np.float64)

        i = 0
        while i < Np_inj:
            random_zeta = self.rng.uniform(1e-15, 1.0 - 1e-15)
            random_beta_x = self.rng.uniform(0.0, 1.0)
            random_beta_y = self.rng.uniform(0.0, 1.0)

            F = emit_x
            nu = emit_x / emit_y

            sigma11_x = emit_x * beta_x_twiss
            sigma12_x = -emit_x * alpha_x_twiss
            sigma22_x = emit_x * gamma_x_twiss

            sigma11_y = emit_y * beta_y_twiss
            sigma12_y = -emit_y * alpha_y_twiss
            sigma22_y = emit_y * gamma_y_twiss

            # https://agenda.linearcollider.org/event/6258/contributions/29168/attachments/24202/37474/linear_dynamics.pdf
            phi_x = 0.5 * np.arctan2(2 * alpha_x_twiss, gamma_x_twiss - beta_x_twiss)
            phi_y = 0.5 * np.arctan2(2 * alpha_y_twiss, gamma_y_twiss - beta_y_twiss)

            X1 = np.sqrt(2) * emit_x / np.sqrt((sigma11_x + sigma22_x) + np.sqrt((sigma22_x - sigma11_x)**2 + 4 * (sigma12_x**2)))
            X2 = np.sqrt(2) * emit_x / np.sqrt((sigma11_x + sigma22_x) - np.sqrt((sigma22_x - sigma11_x)**2 + 4 * (sigma12_x**2)))
            Y1 = np.sqrt(2) * emit_y / np.sqrt((sigma11_y + sigma22_y) + np.sqrt((sigma22_y - sigma11_y)**2 + 4 * (sigma12_y**2)))
            Y2 = np.sqrt(2) * emit_y / np.sqrt((sigma11_y + sigma22_y) - np.sqrt((sigma22_y - sigma11_y)**2 + 4 * (sigma12_y**2)))

            ax = np.sqrt((X1 / X2) * (np.cos(phi_x)**2) + (X2 / X1) * (np.sin(phi_x)**2))
            axpx = (X1 / X2 - X2 / X1) * np.sin(2 * phi_x) / (2 * ax)
            ay = np.sqrt((Y1 / Y2) * (np.cos(phi_y)**2) + (Y2 / Y1) * (np.sin(phi_y)**2))
            aypy = (Y1 / Y2 - Y2 / Y1) * np.sin(2 * phi_y) / (2 * ay)

            zeta_x_square = F * random_zeta
            zeta_x = np.sqrt(zeta_x_square)
            zeta_y_square = (F - zeta_x_square) / nu
            zeta_y = np.sqrt(zeta_y_square)
            beta_x = 2 * const.pi * random_beta_x
            beta_y = 2 * const.pi * random_beta_y

            x = zeta_x * ax * np.cos(beta_x) * 2
            px = zeta_x * (axpx * np.cos(beta_x) - np.sin(beta_x) / ax) * 2
            y = zeta_y * ay * np.cos(beta_y) * 2
            py = zeta_y * (aypy * np.cos(beta_y) - np.sin(beta_y) / ay) * 2

            if x > x_min and x < x_max and y > y_min and y < y_max:
                x_arr[i] = x
                px_arr[i] = px
                y_arr[i] = y
                py_arr[i] = py

                i += 1
            else:
                pass

        p = beam.particles
        p.x[start_index:end_index] = p.xp.asarray(x_arr)
        p.px[start_index:end_index] = p.xp.asarray(px_arr)
        p.y[start_index:end_index] = p.xp.asarray(y_arr)
        p.py[start_index:end_index] = p.xp.asarray(py_arr)

        logger.info(f"Generate successfully")

    def _generate_trans_gaussian_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):

        logger.info(f"The initial transverse Gaussian distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        Np_inj = inj_bunch.Np_inj_curTurn

        emit_x = inj_bunch.emitx
        emit_y = inj_bunch.emity
        alpha_x = inj_bunch.alphax
        alpha_y = inj_bunch.alphay
        beta_x = inj_bunch.betax
        beta_y = inj_bunch.betay
        gamma_x = inj_bunch.gammax
        gamma_y = inj_bunch.gammay

        sigma_x = inj_bunch.sigmax
        sigma_y = inj_bunch.sigmay

        # ε/ε_rms = 1,  [-1 sigma, 1 sigma], the x-px phase space contains: 39.346934029%
        # ε/ε_rms = 2,  [+-sqrt (2 sigma) ], the x-px phase space contains: 63.212055883%
        # ε/ε_rms = 4,  [-2 sigma, 2 sigma], the x-px phase space contains: 86.466471676%
        # ε/ε_rms = 6,  [+-sqrt (6 sigma) ], the x-px phase space contains: 95.021293163%
        # ε/ε_rms = 9,  [-3 sigma, 3 sigma], the x-px phase space contains: 98.889100346%
        # ε/ε_rms = 16, [-4 sigma, 4 sigma], the x-px phase space contains: 99.966453737%
        # ε/ε_rms = 25, [-5 sigma, 5 sigma], the x-px phase space contains: 99.999627335%
        # ε/ε_rms = 36, [-6 sigma, 6 sigma], the x-px phase space contains: 99.999998477%
        x_max = 4 * sigma_x
        x_min = -4 * sigma_x
        y_max = 4 * sigma_y
        y_min = -4 * sigma_y

        x_arr = np.zeros(Np_inj, dtype=np.float64)
        px_arr = np.zeros(Np_inj, dtype=np.float64)
        y_arr = np.zeros(Np_inj, dtype=np.float64)
        py_arr = np.zeros(Np_inj, dtype=np.float64)

        i = 0
        while i < Np_inj:
            random_s1_x = self.rng.uniform(1e-15, 1.0 - 1e-15)
            random_s1_y = self.rng.uniform(1e-15, 1.0 - 1e-15)
            random_s2_x = self.rng.uniform(1e-15, 1.0 - 1e-15)
            random_s2_y = self.rng.uniform(1e-15, 1.0 - 1e-15)

            Xm = 2 * np.sqrt(emit_x * beta_x)
            thetaXm = 2 * np.sqrt(emit_x * gamma_x)
            a_x = np.sqrt(2) / 2 * np.sqrt(-np.log(random_s1_x))
            alp_x = 2 * const.pi * random_s2_x
            chi_x = -1 * np.arctan(alpha_x)
            u_x = a_x * np.cos(alp_x)
            v_x = a_x * np.sin(alp_x)
            x = Xm * u_x
            px = thetaXm * (u_x * np.sin(chi_x) + v_x * np.cos(chi_x))

            Ym = 2 * np.sqrt(emit_y * beta_y)
            thetaYm = 2 * np.sqrt(emit_y * gamma_y)
            a_y = np.sqrt(2) / 2 * np.sqrt(-np.log(random_s1_y))
            alp_y = 2 * const.pi * random_s2_y
            chi_y = -1 * np.arctan(alpha_y)
            u_y = a_y * np.cos(alp_y)
            v_y = a_y * np.sin(alp_y)
            y = Ym * u_y
            py = thetaYm * (u_y * np.sin(chi_y) + v_y * np.cos(chi_y))

            if x > x_min and x < x_max and y > y_min and y < y_max:
                x_arr[i] = x
                px_arr[i] = px
                y_arr[i] = y
                py_arr[i] = py

                i += 1
            else:
                pass

        p = beam.particles
        p.x[start_index:end_index] = p.xp.asarray(x_arr)
        p.px[start_index:end_index] = p.xp.asarray(px_arr)
        p.y[start_index:end_index] = p.xp.asarray(y_arr)
        p.py[start_index:end_index] = p.xp.asarray(py_arr)

        logger.info(f"Generate successfully")

    def _generate_trans_uniform_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):

        logger.info(f"The initial transverse Uniform distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        Np_inj = inj_bunch.Np_inj_curTurn

        emit_x = inj_bunch.emitx
        emit_y = inj_bunch.emity
        alpha_x = inj_bunch.alphax
        alpha_y = inj_bunch.alphay
        beta_x = inj_bunch.betax
        beta_y = inj_bunch.betay
        gamma_x = inj_bunch.gammax
        gamma_y = inj_bunch.gammay

        sigma_x = inj_bunch.sigmax
        sigma_y = inj_bunch.sigmay

        # [-1 sigma, 1 sigma] = 0.6826894921370859, [-4 sigma, 4 sigma] = 0.9999366575163338
        # [-2 sigma, 2 sigma] = 0.9544997361036416, [-5 sigma, 5 sigma] = 0.9999994266968562
        # [-3 sigma, 3 sigma] = 0.9973002039367398, [-6 sigma, 6 sigma] = 0.9999999980268246
        x_max = 4 * sigma_x
        x_min = -4 * sigma_x
        y_max = 4 * sigma_y
        y_min = -4 * sigma_y

        x_arr = np.zeros(Np_inj, dtype=np.float64)
        px_arr = np.zeros(Np_inj, dtype=np.float64)
        y_arr = np.zeros(Np_inj, dtype=np.float64)
        py_arr = np.zeros(Np_inj, dtype=np.float64)

        i = 0
        while i < Np_inj:
            m = 1

            random_s1_x = self.rng.uniform(1e-15, 1.0 - 1e-15)
            random_s1_y = self.rng.uniform(1e-15, 1.0 - 1e-15)
            random_s2_x = self.rng.uniform(1e-15, 1.0 - 1e-15)
            random_s2_y = self.rng.uniform(1e-15, 1.0 - 1e-15)

            Xm = 2 * np.sqrt(emit_x * beta_x)
            thetaXm = 2 * np.sqrt(emit_x * gamma_x)
            Xl = np.sqrt((m + 1) / 2) * Xm
            Xlpx = np.sqrt((m + 1) / 2) * thetaXm
            a_x = np.sqrt(1 - np.power(random_s1_x, 1 / m))
            alp_x = 2 * const.pi * random_s2_x
            chi_x = -1 * np.arctan(alpha_x)
            u_x = a_x * np.cos(alp_x)
            v_x = a_x * np.sin(alp_x)

            x = Xl * u_x
            px = Xlpx * (u_x * np.sin(chi_x) + v_x * np.cos(chi_x))

            Ym = 2 * np.sqrt(emit_y * beta_y)
            thetaYm = 2 * np.sqrt(emit_y * gamma_y)
            Yl = np.sqrt((m + 1) / 2) * Ym
            Ylpy = np.sqrt((m + 1) / 2) * thetaYm
            a_y = np.sqrt(1 - np.power(random_s1_y, 1 / m))
            alp_y = 2 * const.pi * random_s2_y
            chi_y = -1 * np.arctan(alpha_y)
            u_y = a_y * np.cos(alp_y)
            v_y = a_y * np.sin(alp_y)

            y = Yl * u_y
            py = Ylpy * (u_y * np.sin(chi_y) + v_y * np.cos(chi_y))

            if x > x_min and x < x_max and y > y_min and y < y_max:
                x_arr[i] = x
                px_arr[i] = px
                y_arr[i] = y
                py_arr[i] = py

                i += 1
            else:
                pass

        p = beam.particles
        p.x[start_index:end_index] = p.xp.asarray(x_arr)
        p.px[start_index:end_index] = p.xp.asarray(px_arr)
        p.y[start_index:end_index] = p.xp.asarray(y_arr)
        p.py[start_index:end_index] = p.xp.asarray(py_arr)

        logger.info(f"Generate successfully")

    def _generate_longi_gaussian_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):

        logger.info(f"The initial longitudinal Gaussian distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        Np_inj = inj_bunch.Np_inj_curTurn

        sigma_z = inj_bunch.sigmaz
        sigma_pz = inj_bunch.dp

        # [-1 sigma, 1 sigma] = 0.6826894921370859, [-4 sigma, 4 sigma] = 0.9999366575163338
        # [-2 sigma, 2 sigma] = 0.9544997361036416, [-5 sigma, 5 sigma] = 0.9999994266968562
        # [-3 sigma, 3 sigma] = 0.9973002039367398, [-6 sigma, 6 sigma] = 0.9999999980268246
        z_max = 4 * sigma_z
        z_min = -4 * sigma_z

        z_arr = np.zeros(Np_inj, dtype=np.float64)
        pz_arr = np.zeros(Np_inj, dtype=np.float64)

        i = 0
        while i < Np_inj:
            z = self.rng.gauss(0, sigma_z)
            pz = self.rng.gauss(0, sigma_pz)

            if z > z_min and z < z_max:
                z_arr[i] = z
                pz_arr[i] = pz

                i += 1
            else:
                pass

        p = beam.particles
        p.z[start_index:end_index] = p.xp.asarray(z_arr)
        p.pz[start_index:end_index] = p.xp.asarray(pz_arr)

        logger.info(f"Generate successfully")

    def _generate_longi_coasting_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):

        logger.info(f"The initial longitudinal Coasting distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        Np_inj = inj_bunch.Np_inj_curTurn

        sigma_z = inj_bunch.sigmaz  # For costing beam, this is the total length of uniform distribution, not RMS value
        sigma_pz = inj_bunch.dp

        # [-1 sigma, 1 sigma] = 0.6826894921370859, [-4 sigma, 4 sigma] = 0.9999366575163338
        # [-2 sigma, 2 sigma] = 0.9544997361036416, [-5 sigma, 5 sigma] = 0.9999994266968562
        # [-3 sigma, 3 sigma] = 0.9973002039367398, [-6 sigma, 6 sigma] = 0.9999999980268246
        z_max = 0.5 * sigma_z
        z_min = -0.5 * sigma_z

        z_arr = np.zeros(Np_inj, dtype=np.float64)
        pz_arr = np.zeros(Np_inj, dtype=np.float64)

        i = 0
        while i < Np_inj:
            z = self.rng.uniform(-0.5 * sigma_z, 0.5 * sigma_z)
            pz = self.rng.gauss(0, sigma_pz)

            if z > z_min and z < z_max:
                z_arr[i] = z
                pz_arr[i] = pz

                i += 1
            else:
                pass

        p = beam.particles
        p.z[start_index:end_index] = p.xp.asarray(z_arr)
        p.pz[start_index:end_index] = p.xp.asarray(pz_arr)

        logger.info(f"Generate successfully")

    def _generate_longi_matchZ_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):
        # Generate particle's z position and momentum.
        # Use the method in PyHEADTAIL.
        logger.info(f"The initial longitudinal z-matched distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        Np_inj = inj_bunch.Np_inj_curTurn

        sigma_z = inj_bunch.sigmaz
        sigma_pz = inj_bunch.dp

        # [-1 sigma, 1 sigma] = 0.6826894921370859, [-4 sigma, 4 sigma] = 0.9999366575163338
        # [-2 sigma, 2 sigma] = 0.9544997361036416, [-5 sigma, 5 sigma] = 0.9999994266968562
        # [-3 sigma, 3 sigma] = 0.9973002039367398, [-6 sigma, 6 sigma] = 0.9999999980268246
        zmax = inj_bunch.getZMax()
        zmin = inj_bunch.getZMin()
        dp = inj_bunch.getDeltaPMax()
        Hmax = inj_bunch.getHamiltonianPhi(inj_bunch.getUFPPhi(), 0.0)
        H0 = 0.0

        z_arr = np.zeros(Np_inj, dtype=np.float64)
        pz_arr = np.zeros(Np_inj, dtype=np.float64)

        # Check the sigmaz whether the sigmaz > sigma_max
        sigma_max = inj_bunch.getSigmaZ(zmax)
        sig = sigma_z
        # if sigmaz > sigma_max, use sigmaz = 0.99 * sigma_max
        if sig > sigma_max:
            logger.info(f"Sigma z = {sig} is larger than the maximum = {sigma_max}, use the 0.99*sigma_max = {sigma_max*0.99}")
            sig = 0.99 * sigma_max

        # Solve the matched H0
        def func(x):
            return inj_bunch.getSigmaZ(x) - sig

        x2 = sig
        x1 = 0.0
        if func(x2) < 0:
            x1 = sig * 10
        else:
            x1 = sig / 10
        root = brentq(func, x1, x2)
        H0 = inj_bunch.H0FromZ(root)

        i = 0
        while i < Np_inj:

            u = 0.0
            v = 0.0
            s = 0.0

            while True:
                u = self.rng.uniform(0, 1) * (zmax - zmin) + zmin
                v = self.rng.uniform(0, 1) * 2 * dp - dp
                s = self.rng.uniform(0, 1)

                # for stability, limit particles in the 0.9 times bucket
                if s <= inj_bunch.psi(u, v, H0, Hmax) and np.abs(inj_bunch.getHamiltonianZ(u, v)) <= 0.9 * np.abs(Hmax):
                    break

            tmp_z = u
            tmp_pz = v

            if (tmp_z >= (-0.5 * 4 * sigma_z) and tmp_z <= (0.5 * 4 * sigma_z)):
                z_arr[i] = tmp_z
                pz_arr[i] = tmp_pz

                i += 1
            else:
                pass

        p = beam.particles
        p.z[start_index:end_index] = p.xp.asarray(z_arr)
        p.pz[start_index:end_index] = p.xp.asarray(pz_arr)

        logger.info(f"Generate successfully")

    def _generate_longi_matchDp_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):
        # Generate particle's z position and momentum.
        # Use the method in PyHEADTAIL.
        logger.info(f"The initial longitudinal dp-matched distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        Np_inj = inj_bunch.Np_inj_curTurn

        sigma_z = inj_bunch.sigmaz
        sigma_pz = inj_bunch.dp

        # [-1 sigma, 1 sigma] = 0.6826894921370859, [-4 sigma, 4 sigma] = 0.9999366575163338
        # [-2 sigma, 2 sigma] = 0.9544997361036416, [-5 sigma, 5 sigma] = 0.9999994266968562
        # [-3 sigma, 3 sigma] = 0.9973002039367398, [-6 sigma, 6 sigma] = 0.9999999980268246
        zmax = inj_bunch.getZMax()
        zmin = inj_bunch.getZMin()
        dp = inj_bunch.getDeltaPMax()
        Hmax = inj_bunch.getHamiltonianPhi(inj_bunch.getUFPPhi(), 0.0)
        H0 = 0.0

        z_arr = np.zeros(Np_inj, dtype=np.float64)
        pz_arr = np.zeros(Np_inj, dtype=np.float64)

        # Check the sigmaz whether the sigmadp > sigma_max
        sigma_max = inj_bunch.getSigmaDp(dp)
        sig = sigma_pz
        # if sigmaz > sigma_max, use sigmaz = 0.99 * sigma_max
        if sig > sigma_max:
            logger.info(f"Sigma dp = {sig} is larger than the maximum = {sigma_max}, use the 0.99*sigma_max = {sigma_max*0.99}")
            sig = 0.99 * sigma_max

        # Solve the matched H0
        def func(x):
            return inj_bunch.getSigmaDp(x) - sig

        x2 = sig
        x1 = 0.0
        if func(x2) < 0:
            x1 = sig * 10
        else:
            x1 = sig / 10
        root = brentq(func, x1, x2)
        H0 = inj_bunch.H0FromDeltaP(root)

        i = 0
        while i < Np_inj:

            u = 0.0
            v = 0.0
            s = 0.0

            while True:
                u = self.rng.uniform(0, 1) * (zmax - zmin) + zmin
                v = self.rng.uniform(0, 1) * 2 * dp - dp
                s = self.rng.uniform(0, 1)

                # for stability, limit particles in the 0.9 times bucket
                if s <= inj_bunch.psi(u, v, H0, Hmax) and np.abs(inj_bunch.getHamiltonianZ(u, v)) <= 0.9 * np.abs(Hmax):
                    break

            tmp_z = u
            tmp_pz = v

            if (tmp_z >= (-0.5 * 4 * sigma_z) and tmp_z <= (0.5 * 4 * sigma_z)):
                z_arr[i] = tmp_z
                pz_arr[i] = tmp_pz

                i += 1
            else:
                pass

        p = beam.particles
        p.z[start_index:end_index] = p.xp.asarray(z_arr)
        p.pz[start_index:end_index] = p.xp.asarray(pz_arr)

        logger.info(f"Generate successfully")

    def _save_init_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, cfg: Config):

        output_dir = cfg.output_dir_dist
        file_name = f"{cfg.output_hms}_beam{self.beam_id}_bunch{inj_bunch.bunch_id}_{bunch_info.Np}_hor_{inj_bunch.dist_trans}_longi_{inj_bunch.dist_longi}_Dx_{inj_bunch.dx}_injection.tfs"
        file_path = os.path.join(output_dir, file_name)
        logger.info(f"Start saving initial distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} to: {file_path} ...")

        p = beam.particles
        p_cpu = p.copy(np)

        start_index = bunch_info.start_idx
        end_index = bunch_info.end_idx

        df = pd.DataFrame({
            "x": p_cpu.x[start_index:end_index],
            "px": p_cpu.px[start_index:end_index],
            "y": p_cpu.y[start_index:end_index],
            "py": p_cpu.py[start_index:end_index],
            "z": p_cpu.z[start_index:end_index],
            "pz": p_cpu.pz[start_index:end_index],
            "tag": p_cpu.tag[start_index:end_index],
            "lost_turn": p_cpu.lost_turn[start_index:end_index],
            "lost_position": p_cpu.lost_position[start_index:end_index],
            "slice_id": p_cpu.slice_id[start_index:end_index],
        })

        headers = {}
        headers["Name"] = "PASS Distribution Data"
        headers["Transver type"] = inj_bunch.dist_trans
        headers["Longitudinal type"] = inj_bunch.dist_longi
        headers["Beta x"] = inj_bunch.betax
        headers["Beta Y"] = inj_bunch.betay
        headers["Alpha x"] = inj_bunch.alphax
        headers["Alpha y"] = inj_bunch.alphay
        headers["Gamma x"] = inj_bunch.gammax
        headers["Gamma y"] = inj_bunch.gammay
        headers["Emit x"] = inj_bunch.emitx
        headers["Emit y"] = inj_bunch.emity
        headers["Sigma x"] = inj_bunch.sigmax
        headers["Sigma y"] = inj_bunch.sigmay
        headers["Sigma px"] = inj_bunch.sigmapx
        headers["Sigma py"] = inj_bunch.sigmapy
        headers["Sigma z"] = inj_bunch.sigmaz
        headers["Delta p/p"] = inj_bunch.dp
        headers["Turn"] = "Injection"
        headers["Time"] = get_current_time()

        table = tfs.TfsDataFrame(df, headers=headers)
        tfs.write(file_path, table)

        logger.info("Saving successfully")

    def _add_offset(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):
        pass

    def _insert_particles(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):
        pass


class InjectionBunchInfo:

    def __init__(self, beam_id: int, bunch_id: int, sim: Simulation, **kwargs):
        cfg = sim.cfg
        bunch: BunchInfo = sim.beams[beam_id].bunches[bunch_id]

        self.beam_id = beam_id
        self.bunch_id = bunch_id
        self.Ek = bunch.Ek
        self.m0 = bunch.m0
        self.gamma = bunch.gamma
        self.beta = bunch.beta
        self.gamma_t = bunch.gamma_t
        self.circum = bunch.circum
        self.rho = self.circum / (2 * const.pi)
        self.qm_ratio = bunch.qm_ratio
        self.start_turn = 0
        self.stop_turn = int(kwargs["total injection turns"])
        self.interval = int(kwargs["injection interval"])
        self.inj_turns = np.arange(self.start_turn, self.stop_turn, self.interval, dtype=int)
        self.alphax = kwargs["alpha x"]
        self.alphay = kwargs["alpha y"]
        self.betax = kwargs["beta x (m)"]
        self.betay = kwargs["beta y (m)"]
        self.gammax = (1.0 + self.alphax**2) / self.betax
        self.gammay = (1.0 + self.alphay**2) / self.betay
        self.emitx = kwargs["emittance x (m'rad)"]
        self.emity = kwargs["emittance y (m'rad)"]
        self.dx = kwargs["dx (m)"]
        self.dpx = kwargs["dpx"]
        self.sigmax = np.sqrt(self.betax * self.emitx)
        self.sigmay = np.sqrt(self.betay * self.emity)
        self.sigmaz = kwargs["sigma z (m)"]
        self.sigmapx = np.sqrt(self.gammax * self.emitx)
        self.sigmapy = np.sqrt(self.gammay * self.emity)
        self.dp = kwargs["sigma dp/p"]
        self.dist_trans = kwargs["transverse dist"].lower()
        self.dist_longi = kwargs["longitudinal dist"].lower()
        self.rf_voltage = kwargs.get("rf voltage (v)", 0.0)
        self.rf_phi = kwargs.get("rf phase (rad)", 0.0)
        self.harmonic_num = kwargs.get("harmonic number", 0)
        self.harmonic_id = kwargs.get("harmonic id", 0)
        self.rf_position = kwargs.get("rf s position refer to inj. point (m)", 0.0)
        self.is_load_dist = kwargs.get("is load distribution from file", False)
        self.load_dist_filepath = kwargs.get("distribution file path", None)
        self.is_save_init_dist = kwargs.get("is save initial distribution", True)

        self.num_insert_particles = len(kwargs["insert particle coordinate"])
        self.is_insert_particles = self.num_insert_particles > 0
        self.insert_particles = []
        if self.is_insert_particles:
            for i_insert in range(self.num_insert_particles):
                x_tmp = kwargs["insert particle coordinate"][i_insert][0]
                px_tmp = kwargs["insert particle coordinate"][i_insert][1]
                y_tmp = kwargs["insert particle coordinate"][i_insert][2]
                py_tmp = kwargs["insert particle coordinate"][i_insert][3]
                z_tmp = kwargs["insert particle coordinate"][i_insert][4]
                pz_tmp = kwargs["insert particle coordinate"][i_insert][5]

                self.insert_particles.append([x_tmp, px_tmp, y_tmp, py_tmp, z_tmp, pz_tmp])

        kwargs_offset_x = kwargs["offset x"]
        self.is_offset_x = kwargs_offset_x.get("is offset", False)
        if self.is_offset_x:
            self.is_offset_x_fromfile = kwargs_offset_x.get("is load from file", False)
            if self.is_offset_x_fromfile:
                self.offset_x_filepath = kwargs_offset_x["file path"]
                self.offset_x_time, self.offset_x_position, self.offset_x_momentum, self.offset_x_timekind = _read_offset_fromfile(
                    self.offset_x_filepath, "x")
            else:
                self.offset_x_position = np.array([kwargs_offset_x["offset position (m)"]])
                self.offset_x_momentum = np.array([kwargs_offset_x["offset momentum (rad)"]])

        kwargs_offset_y = kwargs["offset y"]
        self.is_offset_y = kwargs_offset_y.get("is offset", False)
        if self.is_offset_y:
            self.is_offset_y_fromfile = kwargs_offset_y.get("is load from file", False)
            if self.is_offset_y_fromfile:
                self.offset_y_filepath = kwargs_offset_y["file path"]
                self.offset_y_time, self.offset_y_position, self.offset_y_momentum, self.offset_y_timekind = _read_offset_fromfile(
                    self.offset_y_filepath, "y")
            else:
                self.offset_y_position = np.array([kwargs_offset_y["offset position (m)"]])
                self.offset_y_momentum = np.array([kwargs_offset_y["offset momentum (rad)"]])

        self.Np_inj_curTurn = 0
        self.Np_injected = 0

    def print(self):
        logger.info(f"\tInjection bunch{self.bunch_id}")
        logger.info(f"\tInjection turns: start={self.start_turn}, stop={self.stop_turn}, interval={self.interval}, total={len(self.inj_turns)}")
        logger.info(
            f"\tTwiss x: alpha={self.alphax:.4f}, beta={self.betax:.4f} m, gamma={self.gammax:.4f} 1/m, emit={self.emitx:.4e} m'rad, sigma={self.sigmax:.4e} m, sigma_px={self.sigmapx:.4e} rad"
        )
        logger.info(
            f"\tTwiss y: alpha={self.alphay:.4f}, beta={self.betay:.4f} m, gamma={self.gammay:.4f} 1/m, emit={self.emity:.4e} m'rad, sigma={self.sigmay:.4e} m, sigma_py={self.sigmapy:.4e} rad"
        )
        logger.info(f"\tDispersion: dx={self.dx:.4f} m, dpx={self.dpx:.4f}")
        logger.info(f"\tLongitudinal: sigma_z={self.sigmaz:.4f} m, sigma_dp/p={self.dp:.4f}")
        logger.info(f"\tDistributions: transverse='{self.dist_trans}', longitudinal='{self.dist_longi}'")
        logger.info(
            f"\tRF: voltage={self.rf_voltage:.2f} V, phase={self.rf_phi:.4f} rad, harmonic_num={self.harmonic_num}, harmonic_id={self.harmonic_id}, s_position={self.rf_position:.4f} m"
        )
        logger.info(f"\tLoad distribution from file: {self.is_load_dist} -> path='{self.load_dist_filepath}'")
        logger.info(f"\tSave initial distribution: {self.is_save_init_dist}")
        logger.info(f"\tInsert particles: num={self.num_insert_particles}, is_insert={self.is_insert_particles}")
        if self.is_insert_particles and self.insert_particles:
            for insert_p in self.insert_particles:
                logger.info(f"\t\t{insert_p}")

        if self.is_offset_x:
            logger.info(f"\tOffset x: enabled, from_file={self.is_offset_x_fromfile}")
            if self.is_offset_x_fromfile:
                logger.info(f"\t\tOffset x file: {self.offset_x_filepath}, time_kind={self.offset_x_timekind}")
                logger.info(
                    f"\t\ttime length={len(self.offset_x_time)}, pos length={len(self.offset_x_position)}, mom length={len(self.offset_x_momentum)}")
            else:
                logger.info(f"\t\tOffset x: position={self.offset_x_position} m, momentum={self.offset_x_momentum} rad")
        else:
            logger.info(f"\tOffset x: disabled")
        if self.is_offset_y:
            logger.info(f"\tOffset y: enabled, from_file={self.is_offset_y_fromfile}")
            if self.is_offset_y_fromfile:
                logger.info(f"\t\tOffset y file: {self.offset_y_filepath}, time_kind={self.offset_y_timekind}")
                logger.info(
                    f"\t\ttime length={len(self.offset_y_time)}, pos length={len(self.offset_y_position)}, mom length={len(self.offset_y_momentum)}")
            else:
                logger.info(f"\t\tOffset y: position={self.offset_y_position} m, momentum={self.offset_y_momentum} rad")
        else:
            logger.info(f"\tOffset y: disabled")

    def phiFromZ(self, z: float):
        return self.rf_phi - self.harmonic_num * z / self.rho

    def zFromPhi(self, phi: float):
        return self.rho * (self.rf_phi - phi) / self.harmonic_num

    def getInitEta(self):
        return 1.0 / self.gamma_t / self.gamma_t - 1.0 / self.gamma / self.gamma

    def getPhiSeparatrix(self, phi: float):
        E = self.Ek + self.m0
        eta = self.getInitEta()
        pi = const.pi
        temp = -1 * self.qm_ratio * self.rf_voltage / pi / self.beta / self.beta / E / self.harmonic_num / eta * (
            np.cos(phi) + np.cos(self.rf_phi) - (pi - phi - self.rf_phi) * np.sin(self.rf_phi))
        if temp < 0:
            temp = 0
        return np.sqrt(temp)

    def getZSeparatrix(self, z: float):
        phi = self.phiFromZ(z)
        return self.getPhiSeparatrix(phi)

    def getUFPPhi(self):
        return const.pi - self.rf_phi

    def getDeltaPMax(self):
        return self.getPhiSeparatrix(self.rf_phi)

    def getPhiMax(self):
        pi = const.pi
        if self.getInitEta() < 0:
            return pi - self.rf_phi
        else:
            phi_syn = self.rf_phi

            def f(x):
                return np.cos(x) + x * np.sin(phi_syn) + np.cos(phi_syn) - (pi - phi_syn) * np.sin(phi_syn)

            root = brentq(f, phi_syn, 2 * pi)
            return root

    def getPhiMin(self):
        pi = const.pi
        if self.getInitEta() > 0:
            return pi - self.rf_phi
        else:
            phi_syn = self.rf_phi

            def f(x):
                return np.cos(x) + x * np.sin(phi_syn) + np.cos(phi_syn) - (pi - phi_syn) * np.sin(phi_syn)

            root = brentq(f, -1 * pi, phi_syn)
            return root

    def getZMax(self):
        phi = self.getPhiMin()
        return self.zFromPhi(phi)

    def getZMin(self):
        phi = self.getPhiMax()
        return self.zFromPhi(phi)

    def getQs(self):
        E = self.Ek + self.m0
        eta = self.getInitEta()
        pi = const.pi
        Qs = np.sqrt(-1 * self.qm_ratio * self.harmonic_num * self.rf_voltage * eta * np.cos(self.rf_phi) / 2 / pi / self.beta / self.beta / E)
        return Qs

    def H0FromZ(self, z: float):
        E = self.Ek + self.m0
        eta = self.getInitEta()
        pi = const.pi
        Qs = self.getQs()
        f0_now = self.beta * const.c / self.circum
        # H0 = -h*2*pi*f0*eta*(vs*z/eta/rho)^2
        H0 = -1 * self.harmonic_num * 2 * pi * f0_now * eta * (Qs * z / eta / self.rho) * (Qs * z / eta / self.rho)
        return H0

    def H0FromDeltaP(self, dp_c: float):
        E = self.Ek + self.m0
        eta = self.getInitEta()
        pi = const.pi
        Qs = self.getQs()
        f0_now = self.beta * const.c / self.circum
        # H0 = -h*2*pi*f0*eta*dp^2
        H0 = -1 * self.harmonic_num * 2 * pi * f0_now * eta * dp_c * dp_c
        return H0

    def getHamiltonianPhi(self, phi: float, deltap: float):
        E = self.Ek + self.m0
        eta = self.getInitEta()
        pi = const.pi
        f0_now = self.beta * const.c / self.circum
        # H = 1/2*h*omega_0*eta*dp^2+omega_0*q*V/2/pi/beta^2/E*(cos(phi)-cos(phi_s)+(phi-phi_s)*sin(phi_s))
        H = (1.0 / 2.0 * self.harmonic_num * 2.0 * pi * f0_now * eta * deltap *
             deltap) + (2.0 * pi * f0_now * self.qm_ratio * self.rf_voltage / 2.0 / pi / self.beta / self.beta / E *
                        (np.cos(phi) - np.cos(self.rf_phi) + (phi - self.rf_phi) * np.sin(self.rf_phi)))
        return H

    def getHamiltonianZ(self, z: float, deltap: float):
        phi = self.phiFromZ(z)
        return self.getHamiltonianPhi(phi, deltap)

    def psi(self, z: float, dp: float, H0: float, Hmax: float):
        # Use the generating function: 1-(exp(H/H0)-1)/(exp(Hmax/H0)-1).
        return 1 - (np.exp(self.getHamiltonianZ(z, dp) / H0) - 1) / (np.exp(Hmax / H0) - 1)

    def getSigmaZ(self, z_c: float):
        zmax = self.getZMax()
        zmin = self.getZMin()

        # Get the separatrix of the buncket
        def dp1(z):
            return -self.getZSeparatrix(z)

        def dp2(z):
            return self.getZSeparatrix(z)

        # Get the H0 and Hmax used in generating function.
        H0 = self.H0FromZ(z_c)
        Hmax = self.getHamiltonianPhi(self.getUFPPhi(), 0.0)
        logger.info(f"H0: {H0}, Hmax: {Hmax}")

        # Get the integral of generating function in the bucket.
        def psi_q(dp, z):
            return self.psi(z, dp, H0, Hmax)

        Q, _ = dblquad(psi_q, zmin, zmax, dp1, dp2)
        logger.info(f"Q: {Q}")

        # Get the mean value of generating function in the bucket.
        def psi_m(dp, z):
            return z * self.psi(z, dp, H0, Hmax)

        M, _ = dblquad(psi_m, zmin, zmax, dp1, dp2)
        M /= Q
        logger.info(f"M: {M}")

        # Get the standard deviation of generating function in the bucket.
        def psi_v(dp, z):
            return (z - M) * (z - M) * self.psi(z, dp, H0, Hmax)

        V, _ = dblquad(psi_v, zmin, zmax, dp1, dp2)
        V /= Q
        logger.info(f"V: {V}")
        return np.sqrt(V)

    def getSigmaDp(self, dp_c: float):
        zmax = self.getZMax()
        zmin = self.getZMin()

        # Get the separatrix of the buncket
        def dp1(z):
            return -self.getZSeparatrix(z)

        def dp2(z):
            return self.getZSeparatrix(z)

        # Get the H0 and Hmax used in generating function.
        H0 = self.H0FromDeltaP(dp_c)
        Hmax = self.getHamiltonianPhi(self.getUFPPhi(), 0.0)

        # Get the integral of generating function in the bucket.
        def psi_q(dp, z):
            return self.psi(z, dp, H0, Hmax)

        Q, _ = dblquad(psi_q, zmin, zmax, dp1, dp2)

        # Get the mean value of generating function in the bucket.
        def psi_m(dp, z):
            return dp * self.psi(z, dp, H0, Hmax)

        M, _ = dblquad(psi_m, zmin, zmax, dp1, dp2)
        M /= Q

        # Get the standard deviation of generating function in the bucket.
        def psi_v(dp, z):
            return (dp - M) * (dp - M) * self.psi(z, dp, H0, Hmax)

        V, _ = dblquad(psi_v, zmin, zmax, dp1, dp2)
        V /= Q

        return np.sqrt(V)


def _read_offset_fromfile(file_path: str, direction: str):
    """
    For offset file, the column names must follow the following rules:

    1. The column name for describing the evolution of time must be 'time (s)' or 'turn'
    
    2. The column name for position must be 'x (m)' or 'y (m)

    3. The column name for momentum must be 'px (rad)' or 'py (rad)
    """

    df = tfs.read(file_path)
    direction = direction.lower()

    time_pattern = re.compile(r'^time\s*(\(\s*s\s*\))?$', re.IGNORECASE)
    turn_pattern = re.compile(r'^turn\s*(\(\s*s\s*\))?$', re.IGNORECASE)

    time_arr = None
    time_kind = None

    for col in df.columns:
        if time_pattern.match(col):
            time_arr = df[col].to_numpy()
            time_kind = 'time'
            break
    if time_arr is None:
        for col in df.columns:
            if turn_pattern.match(col):
                time_arr = df[col].to_numpy()
                time_kind = 'turn'
                break

    if time_arr is None:
        raise KeyError(f"No 'time' or 'turn' columns were found in file: {file_path}.")

    if direction == 'x':
        position_pattern = re.compile(r'^x\s*(\(\s*m\s*\))?$', re.IGNORECASE)
        momentum_pattern = re.compile(r'^px\s*(\(\s*rad\s*\))?$', re.IGNORECASE)
    else:  # direction == 'y'
        position_pattern = re.compile(r'^y\s*(\(\s*m\s*\))?$', re.IGNORECASE)
        momentum_pattern = re.compile(r'^py\s*(\(\s*rad\s*\))?$', re.IGNORECASE)

    position_arr = None
    momentum_arr = None

    for col in df.columns:
        if position_pattern.match(col):
            position_arr = df[col].to_numpy()
        if momentum_pattern.match(col):
            momentum_arr = df[col].to_numpy()
    if position_arr is None:
        raise KeyError(f"No 'x' or 'y' colums were found in file {file_path}")
    if momentum_arr is None:
        raise KeyError(f"No 'px' or 'py' colums were found in file {file_path}")

    return time_arr, position_arr, momentum_arr, time_kind
