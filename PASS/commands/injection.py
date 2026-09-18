from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
import random
import os
from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tfs
from scipy.optimize import brentq
from scipy.integrate import dblquad

from PASS.commands.command import Command
from PASS.core.config import Config
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.bunch import BunchInfo
from PASS.core.particle import ParticlePool
from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.constants import const
from PASS.utils.helper import get_current_time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InjectionBatch:
    """One successfully injected ID interval, independent of current bunch grouping."""

    first_id: int
    count: int
    turn: int
    index: int


class InjectionState:
    """Injection-owned reservations and batch history, attached to the beam.

    Only pending particles need a separate identity map: their tag is zero.
    Once every batch is injected, abs(tag) supplies identity and the map is
    released. Birth information is stored once per batch, never in ParticlePool.
    """

    def __init__(self, count, xp):
        self.remaining = count
        self.reserved_ids = xp.arange(1, count + 1, dtype=xp.int32) if count else None
        self.batches = []

    def reorder(self, permutation):
        if self.reserved_ids is not None:
            self.reserved_ids = self.reserved_ids[permutation]

    def record_batch(self, first_id, count, turn, index):
        self.batches.append(InjectionBatch(first_id, count, turn, index))
        self.remaining -= count
        if self.remaining == 0:
            self.reserved_ids = None

    def snapshot(self, tags):
        """Build optional host-side output columns from signed tags and batch ranges."""
        ids = np.abs(np.asarray(tags))
        turns = np.full(ids.shape, -1, dtype=np.int32)
        indices = np.full(ids.shape, -1, dtype=np.int32)
        batches = sorted(self.batches, key=lambda batch: batch.first_id)
        if batches:
            first = np.array([batch.first_id for batch in batches], dtype=np.int64)
            end = first + np.array([batch.count for batch in batches], dtype=np.int64)
            slot = np.searchsorted(first, ids, side="right") - 1
            valid = (slot >= 0) & (ids < end[np.maximum(slot, 0)])
            turns[valid] = np.array([batch.turn for batch in batches], dtype=np.int32)[slot[valid]]
            indices[valid] = np.array([batch.index for batch in batches], dtype=np.int32)[slot[valid]]
        return {"particle_id": ids, "injection_turn": turns, "injection_batch": indices}


@Command.register("injection")
class Injection(Command):
    """Create incoming batches; lattice elements own transport and material losses."""

    def __init__(self, beam_id: int, sim: Simulation, **command_kwargs):
        kwargs = {k.lower(): v for k, v in command_kwargs.items()}

        self.beam_id = beam_id
        self.s = kwargs["s (m)"]
        self.cmd_type = self.__class__.__name__
        self.cmd_name = kwargs["name"]

        if np.abs(self.s) > const.eps:
            raise ValueError(f"The position s of injection must be 0, but now is {self.s}")

        # The beam harmonic number is declared ONCE at the injection level
        # and defines how many bunch groups are created.  Every bunch dict
        # must be present in the input; declare empty bunches (0 particles)
        # for unfilled groups.
        self.harmonic_number = int(kwargs["harmonic number"])
        self.inj_bunchs = []
        for bunch_id in range(self.harmonic_number):
            b_kwargs = kwargs.get(f"bunch{bunch_id}")
            if b_kwargs is None:
                raise ValueError(f"Injection {self.cmd_name}: bunch{bunch_id} not "
                                 f"declared; harmonic number {self.harmonic_number} "
                                 f"requires {self.harmonic_number} bunch dicts (one per "
                                 f"group).")
            b_kwargs = copy.deepcopy(b_kwargs)
            b_kwargs["harmonic number"] = self.harmonic_number
            inj_bunch = InjectionBunchInfo(self.beam_id, bunch_id, sim, **b_kwargs)
            self.inj_bunchs.append(inj_bunch)

        self.random_seed = kwargs.get("random seed", None)
        if self.random_seed is None:
            self.rng = random.Random()
        else:
            if type(self.random_seed) is not int:
                raise ValueError(f"Injection {self.cmd_name}: random seed must be an integer or null, but now is {self.random_seed}.")
            if self.random_seed < 0:
                logger.warning(
                    f"Injection {self.cmd_name}: random seed must be a non-negative integer, but now is {self.random_seed}. Use its absolute value instead."
                )
                self.random_seed = abs(self.random_seed)
            self.rng = random.Random(self.random_seed)

        self._executed = set()
        self._completed_batches = set()
        self._finished = False
        beam = sim.beams[self.beam_id]
        if getattr(beam, "injection_state", None) is None:
            beam.injection_state = InjectionState(len(beam.particles.tag), beam.particles.xp)
        super().__init__()

    def print(self):
        set_simple_logging()
        logger.info(f"S={self.s:.4f}, Command={self.cmd_type:s}, Name={self.cmd_name:s}, Random Seed={self.random_seed}")
        for inj_bunch in self.inj_bunchs:
            inj_bunch.print()
        set_normal_logging()

    def execute_cpu(self, sim: Simulation):
        return self._execute(sim)

    def execute_gpu(self, sim: Simulation):
        return self._execute(sim)

    def _execute(self, sim: Simulation):
        if self._finished:
            return False
        beam, turn = sim.beams[self.beam_id], int(sim.state.turn)
        if turn in self._executed:
            return False
        p, xp = beam.particles, beam.particles.xp
        state = beam.injection_state
        did_execute = False
        for source in self.inj_bunchs:
            if turn not in source.inj_turns or not source.planned_count:
                continue
            batch = int(np.searchsorted(source.inj_turns, turn))
            count = source.planned_count // len(source.inj_turns)
            if batch == 0:
                count += source.planned_count % len(source.inj_turns)
            if count > 0 and (source.bunch_id, turn) not in self._completed_batches:
                first = source.first_id + source.Np_injected
                if state.reserved_ids is None:
                    raise ValueError("Injection reservations have already been consumed")
                destination = xp.flatnonzero((state.reserved_ids >= first) & (state.reserved_ids < first + count))
                # Keep input row -> ID association, even after SortBunch.
                destination = destination[xp.argsort(state.reserved_ids[destination])]
                if len(destination) != count or bool(xp.any(p.tag[destination] != 0)):
                    raise ValueError("Injection reservation is missing or already active")

                # Generate one batch in isolation. All generators retain their local
                # contiguous-array contract; no circulating particle can be overwritten.
                work = copy.copy(source)
                work.Np_injected, work.Np_inj_curTurn = 0, count
                work.file_start = source.Np_injected
                work.current_turn = turn
                injection_time = beam.reference_program.inverse_integral(float(turn) - source.harmonic_id / source.harmonic_number)
                if source.reference_arrival_time is not None:
                    injection_time += source.reference_arrival_time - beam.reference_program.inverse_integral(
                        -source.harmonic_id / source.harmonic_number)
                work.current_time = injection_time
                batch_particles = ParticlePool(count, np, dtype=p.dtype)
                scratch = SimpleNamespace(particles=batch_particles)
                local = SimpleNamespace(start_idx=0, end_idx=count)
                if work.is_load_dist:
                    self._load_dist(work, local, scratch, True)
                else:
                    transverse = {"kv": "kv", "gaussian": "gaussian", "uniform": "uniform", "waterbag": "waterbag", "parabolic": "parabolic"}
                    longitudinal = {"gaussian": "gaussian", "coasting": "coasting", "matchz": "matchZ", "matchdp": "matchDp"}
                    if work.dist_trans not in transverse or work.dist_longi not in longitudinal:
                        raise ValueError("Unsupported injection distribution")
                    getattr(self, "_generate_trans_" + transverse[work.dist_trans] + "_dist")(work, local, scratch, True)
                    getattr(self, "_generate_longi_" + longitudinal[work.dist_longi] + "_dist")(work, local, scratch, True)
                self._add_offset(work, local, scratch, True)
                if batch == 0 and work.is_insert_particles:
                    if work.num_insert_particles > count:
                        raise ValueError("Manual particles exceed the first injection batch")
                    self._insert_particles(work, local, scratch, True)
                matrix = np.column_stack([getattr(batch_particles, name) for name in ("x", "px", "y", "py", "z", "dp")])
                if not np.all(np.isfinite(matrix)):
                    raise ValueError("Injection coordinates must be finite")
                if np.any((batch_particles.dp <= -1) | ((1 + batch_particles.dp)**2 <= batch_particles.px**2 + batch_particles.py**2)):
                    raise ValueError("Injection requires a real positive longitudinal momentum")

                # The incoming beam keeps its specified energy while the circulating
                # reference may have accelerated. Convert momentum exactly once.
                for bunch in beam.bunches:
                    selected = (destination >= bunch.start_idx) & (destination < bunch.end_idx)
                    dest = destination[selected]
                    rows = np.asarray(selected) if xp is np else selected.get()
                    factor = source.p0 / bunch.p0
                    for name in ("x", "y"):
                        getattr(p, name)[dest] = xp.asarray(getattr(batch_particles, name)[rows])
                    for name in ("px", "py"):
                        getattr(p, name)[dest] = xp.asarray(getattr(batch_particles, name)[rows] * factor)
                    # Keep small momentum deviations when reference momenta are
                    # equal or nearly equal; cast only after the stable transform.
                    incoming_delta = batch_particles.dp[rows].astype(np.float64, copy=False)
                    reference_delta = (source.p0 - bunch.p0) / bunch.p0
                    p.dp[dest] = xp.asarray(incoming_delta * factor + reference_delta)
                    p.z[dest] = xp.asarray(batch_particles.z[rows].astype(np.float64) * (bunch.beta / source.beta) + bunch.beta * const.c *
                                           (bunch.t0 - injection_time))
                p.tag[destination] = state.reserved_ids[destination]
                p.lost_turn[destination], p.lost_position[destination] = -1, -1
                state.record_batch(first, count, turn, batch)
                source.Np_injected += count
                self._completed_batches.add((source.bunch_id, turn))
                source.Np_inj_curTurn = count
                did_execute = True
            # A zero-size final event still owns the requested save. Keep it
            # separate from activation so a failed save can also be retried.
            if (turn == source.inj_turns[-1] and source.is_save_init_dist and not source._saved_init_dist):
                self._save_init_dist(source, beam.bunches[source.bunch_id], beam, sim.cfg)
                source._saved_init_dist = True
                did_execute = True
        self._executed.add(turn)
        self._finished = all(
            source.Np_injected == source.planned_count and (not source.planned_count or not source.is_save_init_dist or source._saved_init_dist)
            for source in self.inj_bunchs)
        return did_execute

    def _load_dist(self, inj_bunch, bunch_info, beam, use_cpu):
        path = Path(inj_bunch.load_dist_filepath)
        df = tfs.read(path)
        fields = ("x", "px", "y", "py", "z", "dp")
        values = df[list(fields)].to_numpy(dtype=float)
        start = 0 if inj_bunch.load_dist_mode == "repeat" else inj_bunch.file_start
        end = start + inj_bunch.Np_inj_curTurn
        if end > len(values):
            raise ValueError(f"Distribution {path} needs {end} rows; found {len(values)}")
        for col, name in enumerate(fields):
            getattr(beam.particles, name)[:] = values[start:end, col]
        beam.particles.dp[:] += inj_bunch.ddp

    def _generate_trans_kv_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):
        # This menthod is derived from "Particle - in - cell code BEAMPATH for beam dynamics simulations in linear accelerators and beamlines"
        # The two beams shoule have different seed values to generate different random values.
        # This is 4-D generator.

        logger.info(f"The initial transverse KV distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        injection_count = inj_bunch.Np_inj_curTurn

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

        x_arr = np.zeros(injection_count, dtype=np.float64)
        px_arr = np.zeros(injection_count, dtype=np.float64)
        y_arr = np.zeros(injection_count, dtype=np.float64)
        py_arr = np.zeros(injection_count, dtype=np.float64)

        # Preserve the original KV map while evaluating its fixed coefficients once.
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
        lower, upper = 1e-15, 1.0 - 1e-15
        boundary_x, boundary_y = 64 * np.spacing(x_max), 64 * np.spacing(y_max)

        i = 0
        while i < injection_count:
            # Draw no more than the missing count, in the original per-candidate order.
            count = min(injection_count - i, 65536)
            values = np.fromiter(iter(self.rng.random, None), dtype=np.float64, count=3 * count).reshape(count, 3)
            random_zeta = lower + (upper - lower) * values[:, 0]
            random_beta_x, random_beta_y = values[:, 1], values[:, 2]
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

            # As in the Gaussian sampler, keep strict-cut decisions on the
            # scalar path when transcendental roundoff could change acceptance.
            near_boundary = ((np.abs(np.abs(x) - x_max) <= boundary_x) | (np.abs(np.abs(y) - y_max) <= boundary_y))
            for row in np.flatnonzero(near_boundary):
                zx2 = F * float(random_zeta[row])
                zx, zy = np.sqrt(zx2), np.sqrt((F - zx2) / nu)
                bx, by = 2 * const.pi * float(random_beta_x[row]), 2 * const.pi * float(random_beta_y[row])
                x[row] = zx * ax * np.cos(bx) * 2
                px[row] = zx * (axpx * np.cos(bx) - np.sin(bx) / ax) * 2
                y[row] = zy * ay * np.cos(by) * 2
                py[row] = zy * (aypy * np.cos(by) - np.sin(by) / ay) * 2
            accepted = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
            end = i + int(np.count_nonzero(accepted))
            x_arr[i:end], px_arr[i:end] = x[accepted], px[accepted]
            y_arr[i:end], py_arr[i:end] = y[accepted], py[accepted]
            i = end

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
        injection_count = inj_bunch.Np_inj_curTurn

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

        x_arr = np.zeros(injection_count, dtype=np.float64)
        px_arr = np.zeros(injection_count, dtype=np.float64)
        y_arr = np.zeros(injection_count, dtype=np.float64)
        py_arr = np.zeros(injection_count, dtype=np.float64)

        # Twiss coefficients are fixed throughout this batch. Keep the scalar
        # formulas and operation order used by the original generator.
        Xm = 2 * np.sqrt(emit_x * beta_x)
        thetaXm = 2 * np.sqrt(emit_x * gamma_x)
        Ym = 2 * np.sqrt(emit_y * beta_y)
        thetaYm = 2 * np.sqrt(emit_y * gamma_y)
        chi_x, chi_y = -np.arctan(alpha_x), -np.arctan(alpha_y)
        sin_chi_x, cos_chi_x = np.sin(chi_x), np.cos(chi_x)
        sin_chi_y, cos_chi_y = np.sin(chi_y), np.cos(chi_y)
        sqrt_half = np.sqrt(2) / 2
        lower, upper = 1e-15, 1.0 - 1e-15
        boundary_x = 64 * np.spacing(x_max)
        boundary_y = 64 * np.spacing(y_max)

        i = 0
        while i < injection_count:
            # At most the missing count: no surplus candidate is discarded,
            # so later longitudinal samples and batches keep their RNG stream.
            count = min(injection_count - i, 65536)
            values = np.fromiter(iter(self.rng.random, None), dtype=np.float64, count=4 * count).reshape(count, 4)
            # random.Random.uniform(a, b) uses a + (b-a)*random(). Columns
            # retain its original per-candidate order: s1x, s1y, s2x, s2y.
            values = lower + (upper - lower) * values
            a_x = sqrt_half * np.sqrt(-np.log(values[:, 0]))
            a_y = sqrt_half * np.sqrt(-np.log(values[:, 1]))
            alp_x = 2 * const.pi * values[:, 2]
            alp_y = 2 * const.pi * values[:, 3]
            u_x, v_x = a_x * np.cos(alp_x), a_x * np.sin(alp_x)
            u_y, v_y = a_y * np.cos(alp_y), a_y * np.sin(alp_y)
            x, y = Xm * u_x, Ym * u_y
            px = thetaXm * (u_x * sin_chi_x + v_x * cos_chi_x)
            py = thetaYm * (u_y * sin_chi_y + v_y * cos_chi_y)

            # Array transcendental functions can round differently from scalar
            # calls. Recheck candidates close to either strict 4-sigma cut with
            # scalar arithmetic, before a changed decision can shift the RNG.
            near_boundary = ((np.abs(np.abs(x) - x_max) <= boundary_x) | (np.abs(np.abs(y) - y_max) <= boundary_y))
            for row in np.flatnonzero(near_boundary):
                s1x, s1y, s2x, s2y = map(float, values[row])
                ax = sqrt_half * np.sqrt(-np.log(s1x))
                ay = sqrt_half * np.sqrt(-np.log(s1y))
                phase_x, phase_y = 2 * const.pi * s2x, 2 * const.pi * s2y
                ux, vx = ax * np.cos(phase_x), ax * np.sin(phase_x)
                uy, vy = ay * np.cos(phase_y), ay * np.sin(phase_y)
                x[row], y[row] = Xm * ux, Ym * uy
                px[row] = thetaXm * (ux * sin_chi_x + vx * cos_chi_x)
                py[row] = thetaYm * (uy * sin_chi_y + vy * cos_chi_y)

            accepted = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
            end = i + int(np.count_nonzero(accepted))
            x_arr[i:end], px_arr[i:end] = x[accepted], px[accepted]
            y_arr[i:end], py_arr[i:end] = y[accepted], py[accepted]
            i = end

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
        injection_count = inj_bunch.Np_inj_curTurn

        emit_x = inj_bunch.emitx
        emit_y = inj_bunch.emity
        alpha_x = inj_bunch.alphax
        alpha_y = inj_bunch.alphay
        beta_x = inj_bunch.betax
        beta_y = inj_bunch.betay
        gamma_x = inj_bunch.gammax
        gamma_y = inj_bunch.gammay

        x_arr = np.zeros(injection_count, dtype=np.float64)
        px_arr = np.zeros(injection_count, dtype=np.float64)
        y_arr = np.zeros(injection_count, dtype=np.float64)
        py_arr = np.zeros(injection_count, dtype=np.float64)

        chi_x = -np.arctan(alpha_x)
        chi_y = -np.arctan(alpha_y)

        Xm = np.sqrt(3.0) * np.sqrt(emit_x * beta_x)
        PXm = np.sqrt(3.0) * np.sqrt(emit_x * gamma_x)
        Ym = np.sqrt(3.0) * np.sqrt(emit_y * beta_y)
        PYm = np.sqrt(3.0) * np.sqrt(emit_y * gamma_y)

        sin_x, cos_x = np.sin(chi_x), np.cos(chi_x)
        sin_y, cos_y = np.sin(chi_y), np.cos(chi_y)
        for start in range(0, injection_count, 65536):
            end = min(start + 65536, injection_count)
            values = np.fromiter(iter(self.rng.random, None), dtype=np.float64, count=4 * (end - start)).reshape(-1, 4)
            # Match uniform(-1, 1), ordered ux, vx, uy, vy for each particle.
            ux, vx, uy, vy = (-1.0 + 2.0 * values).T
            x_arr[start:end] = Xm * ux
            px_arr[start:end] = PXm * (ux * sin_x + vx * cos_x)
            y_arr[start:end] = Ym * uy
            py_arr[start:end] = PYm * (uy * sin_y + vy * cos_y)

        p = beam.particles
        p.x[start_index:end_index] = p.xp.asarray(x_arr)
        p.px[start_index:end_index] = p.xp.asarray(px_arr)
        p.y[start_index:end_index] = p.xp.asarray(y_arr)
        p.py[start_index:end_index] = p.xp.asarray(py_arr)

        logger.info(f"Generate successfully")

    def _generate_trans_waterbag_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):

        logger.info(f"The initial transverse Waterbag distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = (bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn)

        injection_count = inj_bunch.Np_inj_curTurn

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

        x_max = 4.0 * sigma_x
        x_min = -4.0 * sigma_x

        y_max = 4.0 * sigma_y
        y_min = -4.0 * sigma_y

        x_arr = np.zeros(injection_count, dtype=np.float64)
        px_arr = np.zeros(injection_count, dtype=np.float64)

        y_arr = np.zeros(injection_count, dtype=np.float64)
        py_arr = np.zeros(injection_count, dtype=np.float64)

        Xm = np.sqrt(6.0) * np.sqrt(emit_x * beta_x)
        PXm = np.sqrt(6.0) * np.sqrt(emit_x * gamma_x)

        Ym = np.sqrt(6.0) * np.sqrt(emit_y * beta_y)
        PYm = np.sqrt(6.0) * np.sqrt(emit_y * gamma_y)

        chi_x = -np.arctan(alpha_x)
        chi_y = -np.arctan(alpha_y)

        sin_x, cos_x = np.sin(chi_x), np.cos(chi_x)
        sin_y, cos_y = np.sin(chi_y), np.cos(chi_y)
        i = 0
        while i < injection_count:
            count = min(injection_count - i, 65536)
            values = np.fromiter(iter(self.rng.random, None), dtype=np.float64, count=4 * count).reshape(count, 4)
            values = -1.0 + 2.0 * values
            ux, vx, uy, vy = values.T
            # Keep the scalar summation order at the inclusive 4D-ball boundary.
            r2 = ux * ux + vx * vx + uy * uy + vy * vy
            ux, vx, uy, vy = values[r2 <= 1.0].T
            x = Xm * ux
            px = PXm * (ux * sin_x + vx * cos_x)
            y = Ym * uy
            py = PYm * (uy * sin_y + vy * cos_y)
            accepted = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
            end = i + int(np.count_nonzero(accepted))
            x_arr[i:end], px_arr[i:end] = x[accepted], px[accepted]
            y_arr[i:end], py_arr[i:end] = y[accepted], py[accepted]
            i = end

        p = beam.particles

        p.x[start_index:end_index] = p.xp.asarray(x_arr)
        p.px[start_index:end_index] = p.xp.asarray(px_arr)

        p.y[start_index:end_index] = p.xp.asarray(y_arr)
        p.py[start_index:end_index] = p.xp.asarray(py_arr)

        logger.info(f"Generate successfully")

    def _generate_trans_parabolic_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):

        logger.info(f"The initial transverse Parabolic distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = (bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn)

        injection_count = inj_bunch.Np_inj_curTurn

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

        x_max = 4.0 * sigma_x
        x_min = -4.0 * sigma_x

        y_max = 4.0 * sigma_y
        y_min = -4.0 * sigma_y

        x_arr = np.zeros(injection_count, dtype=np.float64)
        px_arr = np.zeros(injection_count, dtype=np.float64)

        y_arr = np.zeros(injection_count, dtype=np.float64)
        py_arr = np.zeros(injection_count, dtype=np.float64)

        Xm = np.sqrt(8.0) * np.sqrt(emit_x * beta_x)
        PXm = np.sqrt(8.0) * np.sqrt(emit_x * gamma_x)

        Ym = np.sqrt(8.0) * np.sqrt(emit_y * beta_y)
        PYm = np.sqrt(8.0) * np.sqrt(emit_y * gamma_y)

        chi_x = -np.arctan(alpha_x)
        chi_y = -np.arctan(alpha_y)

        sin_x, cos_x = np.sin(chi_x), np.cos(chi_x)
        sin_y, cos_y = np.sin(chi_y), np.cos(chi_y)
        draw = self.rng.random
        i = 0
        while i < injection_count:
            count = min(injection_count - i, 65536)
            values = np.empty((count, 4), dtype=np.float64)
            accepted = 0
            while accepted < count:
                # Preserve conditional RNG consumption: the fifth draw occurs
                # only inside the 4D ball, before the next candidate's draws.
                ux = -1.0 + 2.0 * draw()
                vx = -1.0 + 2.0 * draw()
                uy = -1.0 + 2.0 * draw()
                vy = -1.0 + 2.0 * draw()
                r2 = ux * ux + vx * vx + uy * uy + vy * vy
                if r2 <= 1.0 and 0.0 + 1.0 * draw() <= 1.0 - r2:
                    x, y = Xm * ux, Ym * uy
                    if x_min < x < x_max and y_min < y < y_max:
                        values[accepted] = ux, vx, uy, vy
                        accepted += 1
            # Rejection stays scalar; the accepted particles share one array map.
            ux, vx, uy, vy = values.T
            end = i + count
            x_arr[i:end] = Xm * ux
            px_arr[i:end] = PXm * (ux * sin_x + vx * cos_x)
            y_arr[i:end] = Ym * uy
            py_arr[i:end] = PYm * (uy * sin_y + vy * cos_y)
            i = end

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
        injection_count = inj_bunch.Np_inj_curTurn

        sigma_z = inj_bunch.sigmaz
        sigma_dp = inj_bunch.dp

        # [-1 sigma, 1 sigma] = 0.6826894921370859, [-4 sigma, 4 sigma] = 0.9999366575163338
        # [-2 sigma, 2 sigma] = 0.9544997361036416, [-5 sigma, 5 sigma] = 0.9999994266968562
        # [-3 sigma, 3 sigma] = 0.9973002039367398, [-6 sigma, 6 sigma] = 0.9999999980268246
        z_max = 4 * sigma_z
        z_min = -4 * sigma_z

        z_arr = np.zeros(injection_count, dtype=np.float64)
        dp_arr = np.zeros(injection_count, dtype=np.float64)

        i = 0
        while i < injection_count:
            z = self.rng.gauss(0, sigma_z)
            dp = self.rng.gauss(0, sigma_dp)

            if z > z_min and z < z_max:
                z_arr[i] = z
                dp_arr[i] = dp

                i += 1
            else:
                pass

        # Apply bunch-relative longitudinal back-drift from the RF location.
        self._apply_longitudinal_offset(inj_bunch, z_arr, dp_arr)

        p = beam.particles
        p.z[start_index:end_index] = p.xp.asarray(z_arr)
        p.dp[start_index:end_index] = p.xp.asarray(dp_arr)

        logger.info(f"Generate successfully")

    def _generate_longi_coasting_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):

        logger.info(f"The initial longitudinal Coasting distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        injection_count = inj_bunch.Np_inj_curTurn

        sigma_z = inj_bunch.sigmaz  # For costing beam, this is the total length of uniform distribution, not RMS value
        sigma_dp = inj_bunch.dp

        z_max = 0.5 * sigma_z
        z_min = -0.5 * sigma_z

        z_arr = np.zeros(injection_count, dtype=np.float64)
        dp_arr = np.zeros(injection_count, dtype=np.float64)

        i = 0
        while i < injection_count:
            z = self.rng.uniform(-0.5 * sigma_z, 0.5 * sigma_z)
            dp = self.rng.gauss(0, sigma_dp)

            if z > z_min and z < z_max:
                z_arr[i] = z
                dp_arr[i] = dp

                i += 1
            else:
                pass

        # Apply bunch-relative longitudinal back-drift from the RF location.
        self._apply_longitudinal_offset(inj_bunch, z_arr, dp_arr)

        p = beam.particles
        p.z[start_index:end_index] = p.xp.asarray(z_arr)
        p.dp[start_index:end_index] = p.xp.asarray(dp_arr)

        logger.info(f"Generate successfully")

    def _generate_longi_matched_z_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):
        # Generate particle's z position and momentum.
        # Use the method in PyHEADTAIL.
        logger.info(f"The initial longitudinal z-matched distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        injection_count = inj_bunch.Np_inj_curTurn

        sigma_z = inj_bunch.sigmaz
        sigma_dp = inj_bunch.dp

        zmax = inj_bunch.compute_bucket_z_max()
        zmin = inj_bunch.compute_bucket_z_min()
        dp = inj_bunch.compute_bucket_dp_max()
        Hmax = inj_bunch.compute_hamiltonian_from_phase(inj_bunch.compute_unstable_fixed_point_phase(), 0.0)
        H0 = 0.0

        z_arr = np.zeros(injection_count, dtype=np.float64)
        dp_arr = np.zeros(injection_count, dtype=np.float64)

        # Check the sigmaz whether the sigmaz > sigma_max
        sigma_max = inj_bunch.compute_sigma_z(zmax)
        sig = sigma_z
        # if sigmaz > sigma_max, use sigmaz = 0.99 * sigma_max
        if sig > sigma_max:
            logger.info(f"Sigma z = {sig} is larger than the maximum = {sigma_max}, use the 0.99*sigma_max = {sigma_max*0.99}")
            sig = 0.99 * sigma_max

        # Solve the matched H0
        def func(x):
            return inj_bunch.compute_sigma_z(x) - sig

        x2 = sig
        x1 = 0.0
        if func(x2) < 0:
            x1 = sig * 10
        else:
            x1 = sig / 10
        root = brentq(func, x1, x2)
        H0 = inj_bunch.compute_hamiltonian_scale_from_z(root)

        i = 0
        while i < injection_count:

            u = 0.0
            v = 0.0
            s = 0.0

            while True:
                u = self.rng.uniform(0, 1) * (zmax - zmin) + zmin
                v = self.rng.uniform(0, 1) * 2 * dp - dp
                s = self.rng.uniform(0, 1)

                # for stability, limit particles in the 0.9 times bucket
                if s <= inj_bunch.psi(u, v, H0, Hmax) and np.abs(inj_bunch.compute_hamiltonian_from_z(u, v)) <= 0.9 * np.abs(Hmax):
                    break

            sample_z = u
            sample_dp = v

            if (sample_z >= (-0.5 * 4 * sigma_z) and sample_z <= (0.5 * 4 * sigma_z)):
                z_arr[i] = sample_z
                dp_arr[i] = sample_dp

                i += 1
            else:
                pass

        # Apply bunch-relative longitudinal back-drift from the RF location.
        self._apply_longitudinal_offset(inj_bunch, z_arr, dp_arr)

        p = beam.particles
        p.z[start_index:end_index] = p.xp.asarray(z_arr)
        p.dp[start_index:end_index] = p.xp.asarray(dp_arr)

        logger.info(f"Generate successfully")

    def _generate_longi_matched_dp_dist(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):
        # Generate particle's z position and momentum.
        # Use the method in PyHEADTAIL.
        logger.info(f"The initial longitudinal dp-matched distribution of beam{self.beam_id} bunch{inj_bunch.bunch_id} is being generated ...")

        start_index = bunch_info.start_idx + inj_bunch.Np_injected
        end_index = bunch_info.start_idx + inj_bunch.Np_injected + inj_bunch.Np_inj_curTurn
        injection_count = inj_bunch.Np_inj_curTurn

        sigma_z = inj_bunch.sigmaz
        sigma_dp = inj_bunch.dp

        zmax = inj_bunch.compute_bucket_z_max()
        zmin = inj_bunch.compute_bucket_z_min()
        dp = inj_bunch.compute_bucket_dp_max()
        Hmax = inj_bunch.compute_hamiltonian_from_phase(inj_bunch.compute_unstable_fixed_point_phase(), 0.0)
        H0 = 0.0

        z_arr = np.zeros(injection_count, dtype=np.float64)
        dp_arr = np.zeros(injection_count, dtype=np.float64)

        # Check the sigmaz whether the sigmadp > sigma_max
        sigma_max = inj_bunch.compute_sigma_dp(dp)
        sig = sigma_dp
        # if sigmaz > sigma_max, use sigmaz = 0.99 * sigma_max
        if sig > sigma_max:
            logger.info(f"Sigma dp = {sig} is larger than the maximum = {sigma_max}, use the 0.99*sigma_max = {sigma_max*0.99}")
            sig = 0.99 * sigma_max

        # Solve the matched H0
        def func(x):
            return inj_bunch.compute_sigma_dp(x) - sig

        x2 = sig
        x1 = 0.0
        if func(x2) < 0:
            x1 = sig * 10
        else:
            x1 = sig / 10
        root = brentq(func, x1, x2)
        H0 = inj_bunch.compute_hamiltonian_scale_from_dp(root)

        i = 0
        while i < injection_count:

            u = 0.0
            v = 0.0
            s = 0.0

            while True:
                u = self.rng.uniform(0, 1) * (zmax - zmin) + zmin
                v = self.rng.uniform(0, 1) * 2 * dp - dp
                s = self.rng.uniform(0, 1)

                # for stability, limit particles in the 0.9 times bucket
                if s <= inj_bunch.psi(u, v, H0, Hmax) and np.abs(inj_bunch.compute_hamiltonian_from_z(u, v)) <= 0.9 * np.abs(Hmax):
                    break

            sample_z = u
            sample_dp = v

            if (sample_z >= (-0.5 * 4 * sigma_z) and sample_z <= (0.5 * 4 * sigma_z)):
                z_arr[i] = sample_z
                dp_arr[i] = sample_dp

                i += 1
            else:
                pass

        # Apply bunch-relative longitudinal back-drift from the RF location.
        self._apply_longitudinal_offset(inj_bunch, z_arr, dp_arr)

        p = beam.particles
        p.z[start_index:end_index] = p.xp.asarray(z_arr)
        p.dp[start_index:end_index] = p.xp.asarray(dp_arr)

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
            "dp": p_cpu.dp[start_index:end_index],
            "tag": p_cpu.tag[start_index:end_index],
            "lost_turn": p_cpu.lost_turn[start_index:end_index],
            "lost_position": p_cpu.lost_position[start_index:end_index],
        })

        headers = {}
        headers["Name"] = "PASS Distribution Data"
        headers["Trans type"] = inj_bunch.dist_trans
        headers["Longi type"] = inj_bunch.dist_longi
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
        headers["Ek"] = inj_bunch.Ek
        headers["m0"] = inj_bunch.m0
        headers["Harmonic num"] = inj_bunch.harmonic_num
        headers["Rho"] = inj_bunch.rho
        headers["RF voltage"] = inj_bunch.rf_voltage
        headers["RF phase"] = inj_bunch.rf_phi
        headers["Proton num"] = inj_bunch.num_proton
        headers["Neutron num"] = inj_bunch.num_neutron
        headers["Charge num"] = inj_bunch.num_charge
        headers["Gamma T"] = inj_bunch.gamma_t
        headers["Turn"] = "Injection"
        headers["Time"] = get_current_time()

        table = tfs.TfsDataFrame(df, headers=headers)
        tfs.write(file_path, table)

        logger.info("Saving successfully")

    def _apply_longitudinal_offset(self, inj_bunch: InjectionBunchInfo, z_arr: np.ndarray, dp_arr: np.ndarray):
        """Apply longitudinal offset: rf_position back-drift.

        The bunch-relative z is generated around 0 (the bunch center), so no
        bucket shift is applied.  The only correction is the
        back-drift from s_rf to s=0 (reverse propagation over distance
        rf_position): z(s=0) = z(s_rf) + eta * rf_position * dp.

        Parameters
        ----------
        z_arr, dp_arr : ndarray
            Bunch-relative longitudinal coordinates generated around z=0
            (at s=s_rf).  Modified in-place.
        """

        # --- 0. Apply momentum offset (ddp) ---
        # ddp is the bunch-level average momentum deviation, added before
        # back-drift so every dp-dependent correction uses dp + ddp.
        if inj_bunch.ddp != 0.0:
            dp_arr += inj_bunch.ddp

        # rf_position back-drift (reverse propagation s_rf -> s=0).
        # z(s=0) = z(s_rf) - (-1 * eta * rf_position * dp)
        eta = inj_bunch.compute_initial_slip_factor()
        z_arr += eta * inj_bunch.rf_position * dp_arr

    def _add_offset(self, inj_bunch, bunch_info, beam, use_cpu):
        p = beam.particles
        start = bunch_info.start_idx + inj_bunch.Np_injected
        end = start + inj_bunch.Np_inj_curTurn
        bunch_slice = slice(start, end)
        p.x[bunch_slice] += inj_bunch.dx * p.dp[bunch_slice]
        p.px[bunch_slice] += inj_bunch.dpx * p.dp[bunch_slice]
        for axis in ("x", "y"):
            if not getattr(inj_bunch, "is_offset_" + axis, False):
                continue
            position = getattr(inj_bunch, "offset_" + axis + "_position")
            momentum = getattr(inj_bunch, "offset_" + axis + "_momentum")
            if getattr(inj_bunch, "is_offset_" + axis + "_fromfile"):
                nodes = getattr(inj_bunch, "offset_" + axis + "_time")
                kind = getattr(inj_bunch, "offset_" + axis + "_timekind")
                when = inj_bunch.current_turn if kind == "turn" else inj_bunch.current_time
                if when < nodes[0] or when > nodes[-1]:
                    raise ValueError(f"Injection offset {axis} does not cover {kind}={when}")
                position, momentum = np.interp(when, nodes, position), np.interp(when, nodes, momentum)
            else:
                position, momentum = position[0], momentum[0]
            getattr(p, axis)[bunch_slice] += position
            getattr(p, "p" + axis)[bunch_slice] += momentum

    def _insert_particles(self, inj_bunch: InjectionBunchInfo, bunch_info: BunchInfo, beam: Beam, use_cpu: bool):
        logger.info(f"Inserting specified particles to beam{self.beam_id} bunch{inj_bunch.bunch_id} ...")

        num_insert_particles = inj_bunch.num_insert_particles
        insert_particles = inj_bunch.insert_particles

        start_index = bunch_info.start_idx
        end_index = bunch_info.start_idx + num_insert_particles
        injection_count = inj_bunch.Np_inj_curTurn

        insert_arr = np.asarray(insert_particles)

        x_arr = insert_arr[:, 0]
        px_arr = insert_arr[:, 1]
        y_arr = insert_arr[:, 2]
        py_arr = insert_arr[:, 3]
        z_arr = insert_arr[:, 4]
        dp_arr = insert_arr[:, 5]

        p = beam.particles

        p.x[start_index:end_index] = p.xp.asarray(x_arr)
        p.px[start_index:end_index] = p.xp.asarray(px_arr)
        p.y[start_index:end_index] = p.xp.asarray(y_arr)
        p.py[start_index:end_index] = p.xp.asarray(py_arr)
        p.z[start_index:end_index] = p.xp.asarray(z_arr)
        p.dp[start_index:end_index] = p.xp.asarray(dp_arr)

        logger.info(f"Insert successfully")


class InjectionBunchInfo:

    def __init__(self, beam_id: int, bunch_id: int, sim: Simulation, **kwargs):
        cfg = sim.cfg
        bunch: BunchInfo = sim.beams[beam_id].bunches[bunch_id]

        self.planned_count = bunch.Np
        self.first_id = bunch.start_idx + 1
        self.p0 = bunch.p0
        self.reference_arrival_time = kwargs.get("reference arrival time (s)")
        self.harmonic_id = bunch.harmonic_id
        self.harmonic_number = bunch.harmonic_number
        self.beam_id = beam_id
        self.bunch_id = bunch_id
        self.Ek = bunch.Ek
        self.m0 = bunch.m0
        self.gamma = bunch.gamma
        self.beta = bunch.beta
        self.gamma_t = bunch.gamma_t
        self.circum = bunch.circum
        self.rho = self.circum / (2 * const.pi)
        self.num_proton = bunch.num_proton
        self.num_neutron = bunch.num_neutron
        self.num_charge = bunch.num_charge
        self.qm_ratio = bunch.qm_ratio
        self.start_turn = 0
        self.stop_turn = int(kwargs["total injection turns"])
        self.interval = int(kwargs["injection interval"])
        if self.stop_turn < 1 or self.interval < 1:
            raise ValueError("Injection turns and interval must be positive")
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
        self.harmonic_id = kwargs.get("harmonic id of this bunch", 0)
        self.rf_position = kwargs.get("rf s position refer to inj. point (m)", 0.0)

        # Momentum offset: support both dp offset and kinetic energy offset
        ddp = kwargs.get("momentum offset dp", 0.0)
        dde = kwargs.get("kinetic energy offset (ev)", 0.0)
        if ddp != 0.0 and dde != 0.0:
            raise ValueError(f"Bunch{bunch_id}: 'momentum offset dp' and 'kinetic energy offset (eV)' "
                             f"are mutually exclusive, please set only one to non-zero.")
        if dde != 0.0:
            # Convert kinetic energy offset to dp offset using exact E^2 = p^2 + m_0^2
            # E0 = Ek + m0, p0 = sqrt(E0^2 - m0^2)
            # E1 = E0 + dde, p1 = sqrt(E1^2 - m0^2)
            # dp = p1/p0 - 1
            E0 = self.Ek + self.m0
            p0 = self.gamma * self.m0 * self.beta
            E1 = E0 + dde
            p1 = np.sqrt(E1 * E1 - self.m0 * self.m0)
            ddp = p1 / p0 - 1.0
        self.ddp = ddp
        self.is_load_dist = kwargs.get("is load distribution from file", False)
        self.load_dist_filepath = kwargs.get("distribution file path", None)
        self.load_dist_mode = kwargs.get("distribution file mode", "sequential")
        if self.load_dist_mode not in {"sequential", "repeat"}:
            raise ValueError("Distribution File Mode must be sequential or repeat")
        self.is_save_init_dist = kwargs.get("is save initial distribution", True)
        self._saved_init_dist = False

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
                dp_tmp = kwargs["insert particle coordinate"][i_insert][5]

                self.insert_particles.append([x_tmp, px_tmp, y_tmp, py_tmp, z_tmp, dp_tmp])

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

    def compute_phase_from_z(self, z: float):
        return self.rf_phi - self.harmonic_num * z / self.rho

    def compute_z_from_phase(self, phi: float):
        return self.rho * (self.rf_phi - phi) / self.harmonic_num

    def compute_initial_slip_factor(self):
        return 1.0 / self.gamma_t / self.gamma_t - 1.0 / self.gamma / self.gamma

    def compute_separatrix_dp_from_phase(self, phi: float):
        E = self.Ek + self.m0
        eta = self.compute_initial_slip_factor()
        pi = const.pi
        temp = -1 * self.qm_ratio * self.rf_voltage / pi / self.beta / self.beta / E / self.harmonic_num / eta * (
            np.cos(phi) + np.cos(self.rf_phi) - (pi - phi - self.rf_phi) * np.sin(self.rf_phi))
        if temp < 0:
            temp = 0
        return np.sqrt(temp)

    def compute_separatrix_dp_from_z(self, z: float):
        phi = self.compute_phase_from_z(z)
        return self.compute_separatrix_dp_from_phase(phi)

    def compute_unstable_fixed_point_phase(self):
        return const.pi - self.rf_phi

    def compute_bucket_dp_max(self):
        return self.compute_separatrix_dp_from_phase(self.rf_phi)

    def compute_bucket_phase_max(self):
        pi = const.pi
        if self.compute_initial_slip_factor() < 0:
            return pi - self.rf_phi
        else:
            phi_syn = self.rf_phi

            def f(x):
                return np.cos(x) + x * np.sin(phi_syn) + np.cos(phi_syn) - (pi - phi_syn) * np.sin(phi_syn)

            root = brentq(f, phi_syn, 2 * pi)
            return root

    def compute_bucket_phase_min(self):
        pi = const.pi
        if self.compute_initial_slip_factor() > 0:
            return pi - self.rf_phi
        else:
            phi_syn = self.rf_phi

            def f(x):
                return np.cos(x) + x * np.sin(phi_syn) + np.cos(phi_syn) - (pi - phi_syn) * np.sin(phi_syn)

            root = brentq(f, -1 * pi, phi_syn)
            return root

    def compute_bucket_z_max(self):
        phi = self.compute_bucket_phase_min()
        return self.compute_z_from_phase(phi)

    def compute_bucket_z_min(self):
        phi = self.compute_bucket_phase_max()
        return self.compute_z_from_phase(phi)

    def compute_synchrotron_tune(self):
        E = self.Ek + self.m0
        eta = self.compute_initial_slip_factor()
        pi = const.pi
        Qs = np.sqrt(-1 * self.qm_ratio * self.harmonic_num * self.rf_voltage * eta * np.cos(self.rf_phi) / 2 / pi / self.beta / self.beta / E)
        return Qs

    def compute_hamiltonian_scale_from_z(self, z: float):
        E = self.Ek + self.m0
        eta = self.compute_initial_slip_factor()
        pi = const.pi
        Qs = self.compute_synchrotron_tune()
        f0_now = self.beta * const.c / self.circum
        # H0 = -h*2*pi*f0*eta*(vs*z/eta/rho)^2
        H0 = -1 * self.harmonic_num * 2 * pi * f0_now * eta * (Qs * z / eta / self.rho) * (Qs * z / eta / self.rho)
        return H0

    def compute_hamiltonian_scale_from_dp(self, dp_c: float):
        E = self.Ek + self.m0
        eta = self.compute_initial_slip_factor()
        pi = const.pi
        Qs = self.compute_synchrotron_tune()
        f0_now = self.beta * const.c / self.circum
        # H0 = -h*2*pi*f0*eta*dp^2
        H0 = -1 * self.harmonic_num * 2 * pi * f0_now * eta * dp_c * dp_c
        return H0

    def compute_hamiltonian_from_phase(self, phi: float, deltap: float):
        E = self.Ek + self.m0
        eta = self.compute_initial_slip_factor()
        pi = const.pi
        f0_now = self.beta * const.c / self.circum
        # H = 1/2*h*omega_0*eta*dp^2+omega_0*q*V/2/pi/beta^2/E*(cos(phi)-cos(phi_s)+(phi-phi_s)*sin(phi_s))
        H = (1.0 / 2.0 * self.harmonic_num * 2.0 * pi * f0_now * eta * deltap *
             deltap) + (2.0 * pi * f0_now * self.qm_ratio * self.rf_voltage / 2.0 / pi / self.beta / self.beta / E *
                        (np.cos(phi) - np.cos(self.rf_phi) + (phi - self.rf_phi) * np.sin(self.rf_phi)))
        return H

    def compute_hamiltonian_from_z(self, z: float, deltap: float):
        phi = self.compute_phase_from_z(z)
        return self.compute_hamiltonian_from_phase(phi, deltap)

    def psi(self, z: float, dp: float, H0: float, Hmax: float):
        # Use the generating function: 1-(exp(H/H0)-1)/(exp(Hmax/H0)-1).
        return 1 - (np.exp(self.compute_hamiltonian_from_z(z, dp) / H0) - 1) / (np.exp(Hmax / H0) - 1)

    def compute_sigma_z(self, z_c: float):
        zmax = self.compute_bucket_z_max()
        zmin = self.compute_bucket_z_min()

        # Get the separatrix of the buncket
        def dp1(z):
            return -self.compute_separatrix_dp_from_z(z)

        def dp2(z):
            return self.compute_separatrix_dp_from_z(z)

        # Get the H0 and Hmax used in generating function.
        H0 = self.compute_hamiltonian_scale_from_z(z_c)
        Hmax = self.compute_hamiltonian_from_phase(self.compute_unstable_fixed_point_phase(), 0.0)

        # Get the integral of generating function in the bucket.
        def psi_q(dp, z):
            return self.psi(z, dp, H0, Hmax)

        Q, _ = dblquad(psi_q, zmin, zmax, dp1, dp2)

        # Get the mean value of generating function in the bucket.
        def psi_m(dp, z):
            return z * self.psi(z, dp, H0, Hmax)

        M, _ = dblquad(psi_m, zmin, zmax, dp1, dp2)
        M /= Q

        # Get the standard deviation of generating function in the bucket.
        def psi_v(dp, z):
            return (z - M) * (z - M) * self.psi(z, dp, H0, Hmax)

        V, _ = dblquad(psi_v, zmin, zmax, dp1, dp2)
        V /= Q
        return np.sqrt(V)

    def compute_sigma_dp(self, dp_c: float):
        zmax = self.compute_bucket_z_max()
        zmin = self.compute_bucket_z_min()

        # Get the separatrix of the buncket
        def dp1(z):
            return -self.compute_separatrix_dp_from_z(z)

        def dp2(z):
            return self.compute_separatrix_dp_from_z(z)

        # Get the H0 and Hmax used in generating function.
        H0 = self.compute_hamiltonian_scale_from_dp(dp_c)
        Hmax = self.compute_hamiltonian_from_phase(self.compute_unstable_fixed_point_phase(), 0.0)

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

    if (len(time_arr) == 0 or not np.all(np.isfinite(time_arr)) or np.any(np.diff(time_arr) <= 0) or not np.all(np.isfinite(position_arr))
            or not np.all(np.isfinite(momentum_arr))):
        raise ValueError("Injection offset requires finite values and strictly increasing nodes")
    return time_arr, position_arr, momentum_arr, time_kind
