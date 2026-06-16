from __future__ import annotations

from PASS.commands.command import Command
from PASS.core.config import Config
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const

import numpy as np
import logging
import tfs
import re
from pathlib import Path

logger = logging.getLogger(__name__)


@Command.register("injection")
class Injection(Command):

    def __init__(self, beam_id: int, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        if np.abs(self.s) > const.eps:
            raise ValueError(f"The position s of injection must be 0, but now is {self.s}")

        self.num_bunch = len(kwargs) - 2
        self.inj_bunchs = []
        for i in range(self.num_bunch):
            inj_bunch = InjectionBunchInfo(i, **kwargs[f"bunch{i}"])
            self.inj_bunchs.append(inj_bunch)

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

        turn = state.turn

        for i in range(beam.num_bunch):
            inj_bunch = self.inj_bunchs[i]
            if turn not in inj_bunch.inj_turns:
                return
            bunch_info = bunches[i]
            Np = bunch_info.Np
            total_inj_turns = len(inj_bunch.inj_turns)
            inj_bunch.Np_inj_curTurn = int(Np / total_inj_turns)
            logger.info(f"total injection turns = {total_inj_turns}, Np_inj_curTurn = {inj_bunch.Np_inj_curTurn}")
            if (turn == 0 and inj_bunch.Np_inj_curTurn * total_inj_turns != Np):
                inj_bunch.Np_inj_curTurn += (Np - inj_bunch.Np_inj_curTurn * total_inj_turns)
                logger.info(
                    f"[Injection] Since the total number of particles {Np} cannot be divided exactly by the number of injection turns {total_inj_turns}, we will inject {inj_bunch.Np_inj_curTurn} particles in the first turn and {Np / total_inj_turns} particles in the rest turns."
                )

            if inj_bunch.is_load_dist:
                _load_dist(inj_bunch, bunch_info, beam)
            else:
                if inj_bunch.dist_trans.lower() == "kv":
                    pass
                elif inj_bunch.dist_trans.lower() == "gaussian":
                    pass
                elif inj_bunch.dist_trans.lower() == "uniform":
                    pass
                else:
                    raise ValueError(f"We don't support transverse distribution: {inj_bunch.dist_trans}")

                if inj_bunch.dist_longi.lower() == "gaussian":
                    pass
                elif inj_bunch.dist_longi.lower() == "coasting":
                    pass
                elif inj_bunch.dist_longi.lower() == "matchZ":
                    pass
                elif inj_bunch.dist_longi.lower() == "matchDp":
                    pass
                else:
                    raise ValueError(f"We don't support longitudinal distribution: {inj_bunch.dist_longi}")

            # add_offset()

            inj_bunch.Np_injected += inj_bunch.Np_inj_curTurn

            if turn == inj_bunch.inj_turns[0]:
                if inj_bunch.is_insert_particles:
                    pass

            if turn == inj_bunch.inj_turns[-1]:
                if inj_bunch.is_save_init_dist:
                    pass


def _load_dist(inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):
    path = Path(inj_bunch.load_dist_filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = tfs.read(path)
    x = df["x"].to_numpy()
    px = df["px"].to_numpy()
    y = df["y"].to_numpy()
    py = df["py"].to_numpy()
    z = df["z"].to_numpy()
    pz = df["pz"].to_numpy()

    len_input = len(x)

    start_index = inj_bunch.Np_injected
    end_index = inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn

    copy_start = start_index
    copy_end = min(end_index, len_input)

    if use_cpu:
        xp = np
    else:
        xp = cp

    if copy_start >= len_input:
        logger.warning(f"No more particles to inject from file {path}. Start index {copy_start} beyond file length {len_input}")
    else:
        df_start = copy_start
        df_end = copy_end

        p = beam.particles
        p.x[copy_start:copy_end] = xp.asarray(x[df_start:df_end])
        p.px[copy_start:copy_end] = xp.asarray(px[df_start:df_end])
        p.y[copy_start:copy_end] = xp.asarray(y[df_start:df_end])
        p.py[copy_start:copy_end] = xp.asarray(py[df_start:df_end])
        p.z[copy_start:copy_end] = xp.asarray(z[df_start:df_end])
        p.pz[copy_start:copy_end] = xp.asarray(pz[df_start:df_end])

        if copy_end < end_index:
            logger.warning(f"Only copy particles {copy_start}-{copy_end} from file: {path}")


class InjectionBunchInfo:

    def __init__(self, bunch_id: int, **kwargs):
        self.bunch_id = bunch_id
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
                x_tmp = kwargs["insert particle coordinate"][0][0]
                px_tmp = kwargs["insert particle coordinate"][0][1]
                y_tmp = kwargs["insert particle coordinate"][0][2]
                py_tmp = kwargs["insert particle coordinate"][0][3]
                z_tmp = kwargs["insert particle coordinate"][0][4]
                pz_tmp = kwargs["insert particle coordinate"][0][5]

                self.insert_particles.append([x_tmp, px_tmp, y_tmp, py_tmp, z_tmp, pz_tmp])

        kwargs_offset_x = kwargs["offset x"]
        self.is_offset_x = kwargs_offset_x.get("is offset", False)
        if self.is_offset_x:
            self.is_offset_x_fromfile = kwargs_offset_x.get("is load from file", False)
            if self.is_offset_x_fromfile:
                self.offset_x_filepath = kwargs_offset_x["file path"]
                self.offset_x_time, self.offset_x_position, self.offset_x_momentum, self.offset_x_timekind = _read_offset_fromfile(
                    self.offset_x_filepath)
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
                    self.offset_y_filepath)
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
                logger.info(f"\t\tOffset x: position={self.offset_x_position} m, momentum={self.offset_x_momemtum} rad")
        else:
            logger.info(f"\tOffset x: disabled")
        if self.is_offset_y:
            logger.info(f"\tOffset y: enabled, from_file={self.is_offset_y_fromfile}")
            if self.is_offset_y_fromfile:
                logger.info(f"\t\tOffset y file: {self.offset_y_filepath}, time_kind={self.offset_y_timekind}")
                logger.info(
                    f"\t\ttime length={len(self.offset_y_time)}, pos length={len(self.offset_y_position)}, mom length={len(self.offset_y_momentum)}")
            else:
                logger.info(f"\t\tOffset y: position={self.offset_y_position} m, momentum={self.offset_y_momemtum} rad")
        else:
            logger.info(f"\tOffset y: disabled")


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
