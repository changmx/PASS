"""Independent bunch-centroid FIR feedback, with CPU and batched CUDA tracking.

The signal is a position in metres; gain converts it to P_transverse/P_reference.
The fixed filter is evaluated at the pickup and queued for an explicit turn.
"""

import csv
import hashlib
import logging
from pathlib import Path
import re
import weakref

import numpy as np

from PASS.commands.command import Command
from PASS.para.schema.transverse_feedback import TransverseFeedbackItem, TransversePickupItem


def _parse_parameters(model, command_kwargs):
    kwargs = {k: v for k, v in command_kwargs.items() if k.lower() != "name"}
    return model.model_validate(kwargs)


def _bind_feedback_pair(beam, command, name, index):
    """Command-owned references allow either endpoint to be constructed first."""
    # Config lowercases sequence keys but preserves string-valued references.
    name = name.lower()
    pairs = getattr(beam, "_transverse_feedback_pairs", None)
    if pairs is None:
        pairs = {}
        beam._transverse_feedback_pairs = pairs
    pair = pairs.setdefault(name, [None, None])
    if pair[index] is not None:
        raise ValueError(f"Each transverse pickup must have exactly one feedback: {name!r}")
    pair[index] = command
    pickup, feedback = pair
    if pickup is not None and feedback is not None:
        pickup.feedback = feedback
        feedback.pickup = pickup


@Command.register("transversepickup")
class TransversePickup(Command):
    """Non-invasive position pickup for one named transverse feedback loop."""

    def __init__(self, beam_id, sim, **command_kwargs):
        self.parameters = _parse_parameters(TransversePickupItem, command_kwargs)
        self.beam_id = beam_id
        self.cmd_name = next(v for k, v in command_kwargs.items() if k.lower() == "name")
        self.cmd_type = type(self).__name__
        self.s = self.parameters.s
        self.length = 0.0
        self.order = self.parameters.order
        self.feedback = None
        _bind_feedback_pair(sim.beams[beam_id], self, self.cmd_name, 0)

    def print(self):
        logging.getLogger(__name__).info("S=%g, Command=%s, Name=%s, Plane=%s", self.s, self.cmd_type, self.cmd_name, self.parameters.plane)

    def execute_cpu(self, sim):
        if self.feedback is None:
            raise ValueError(f"TransversePickup {self.cmd_name!r} requires one feedback in the same beam")
        return self.feedback._sample_cpu(int(sim.state.turn))

    def execute_gpu(self, sim):
        if self.feedback is None:
            raise ValueError(f"TransversePickup {self.cmd_name!r} requires one feedback in the same beam")
        return self.feedback._sample_gpu(int(sim.state.turn))


@Command.register("transversefeedback")
class TransverseFeedback(Command):
    """Apply a delayed, bunch-specific normalized momentum kick.

    All state arrays live on the particle backend. Explicit inspection and
    checkpoint operations may transfer them to the host; tracking does not.
    """

    def __init__(self, beam_id, sim, **command_kwargs):
        self.parameters = _parse_parameters(TransverseFeedbackItem, command_kwargs)
        self.beam_id = beam_id
        self.cmd_name = next(v for k, v in command_kwargs.items() if k.lower() == "name")
        self.cmd_type = type(self).__name__
        self.s = self.parameters.s
        self.length = 0.0
        self.order = self.parameters.order
        self.is_enabled = self.parameters.enabled
        self.start_turn = self.parameters.start_turn
        self.end_turn = self.parameters.end_turn if self.parameters.end_turn is not None else sim.cfg.num_turn
        self.delay_turns = self.parameters.delay_turns
        self.pickup = None
        self._beam = sim.beams[beam_id]
        self._cfg = sim.cfg
        self._prepared = False
        self._layout_key = None
        self._layout_tag = None
        self._particle_array_refs = None
        self._last_pickup_turn = -1
        self._last_kick_turn = -1
        self._diagnostic_size = 0
        self._diagnostic_path = None
        _bind_feedback_pair(self._beam, self, self.parameters.pickup, 1)
        observers = getattr(self._beam, "_bunch_injection_observers", None)
        if observers is None:
            observers = []
            self._beam._bunch_injection_observers = observers
        observers.append(self._on_injection)

    def print(self):
        logging.getLogger(__name__).info("S=%g, Command=%s, Name=%s, Pickup=%s, Delay=%d turns", self.s, self.cmd_type, self.cmd_name,
                                         self.parameters.pickup, self.delay_turns)

    def _ensure_prepared(self):
        if not self._prepared:
            if self.pickup is None:
                raise ValueError(f"TransverseFeedback requires pickup {self.parameters.pickup!r} in the same beam")
            self._initialize(self.pickup)

    def _initialize(self, pickup):
        if self._prepared:
            if self.pickup is not pickup:
                raise ValueError("A prepared feedback cannot be rebound to another pickup")
            return
        self.pickup = pickup
        if any(max(pickup.s, self.s) > bunch.circum for bunch in self._beam.bunches):
            raise ValueError("Feedback nodes must lie within the ring circumference")
        self.plane_mask = {"x": 1, "y": 2, "xy": 3}[pickup.parameters.plane]
        p = self._beam.particles
        self.xp = p.xp
        self._particle_dtype = p.dtype
        self._device_id = None if self.xp is np else p.x.device.id
        self.n_bunches = int(self._beam.harmonic_number)
        coefficients = (self.parameters.coefficients_x, self.parameters.coefficients_y)
        for plane, values in enumerate(coefficients):
            if self.plane_mask & (1 << plane):
                if values is None:
                    raise ValueError("Each selected pickup plane requires FIR coefficients")
            elif values is not None:
                raise ValueError("FIR coefficients were supplied for an unmeasured plane")
        if (not self.plane_mask & 1 and self.parameters.gain_x != 0) or (not self.plane_mask & 2 and self.parameters.gain_y != 0):
            raise ValueError("Gain for an unmeasured plane must be zero")
        self.n_taps = max(len(values) for values in coefficients if values is not None)
        coeff = np.zeros((self.n_taps, 2), dtype=np.float64)
        for plane, values in enumerate(coefficients):
            if values is not None:
                coeff[:len(values), plane] = values
        active = np.zeros(self.n_bunches, dtype=np.int32)
        selected = self.parameters.bunch_ids
        if selected is not None and any(slot >= self.n_bunches for slot in selected):
            raise ValueError("Feedback Harmonic IDs must belong to the injection grouping")
        active[:] = 1 if selected is None else 0
        if selected is not None:
            active[selected] = 1
        reference = np.tile([pickup.parameters.reference_x, pickup.parameters.reference_y], (self.n_bunches, 1))
        gain = np.tile([self.parameters.gain_x, self.parameters.gain_y], (self.n_bunches, 1))
        limits = [self.parameters.max_kick_x, self.parameters.max_kick_y]
        max_kick = np.tile([np.inf if value is None else value for value in limits], (self.n_bunches, 1))
        self.coefficients = self.xp.asarray(coeff)
        self.active = self.xp.asarray(active)
        self.reference = self.xp.asarray(reference, dtype=np.float64)
        self.gain = self.xp.asarray(gain, dtype=np.float64)
        self.max_kick = self.xp.asarray(max_kick, dtype=np.float64)
        self._active_host = active
        self._generation_host = np.zeros(self.n_bunches, dtype=np.int64)
        self.history = self.xp.zeros((self.n_taps, self.n_bunches, 2), dtype=np.float64)
        self.history_turn = self.xp.full((self.n_taps, self.n_bunches), -1, dtype=np.int64)
        self.pending = self.xp.zeros((self.delay_turns + 1, self.n_bunches, 2), dtype=np.float64)
        self.pending_turn = self.xp.full((self.delay_turns + 1, self.n_bunches), -1, dtype=np.int64)
        self.generation = self.xp.zeros(self.n_bunches, dtype=np.int64)
        self.seen_generation = self.xp.zeros(self.n_bunches, dtype=np.int64)
        self.latest = self.xp.zeros((self.n_bunches, 10), dtype=np.float64)
        self.latest[:, 9] = -1
        self._refresh_layout()
        if self.xp is not np:
            self._compile_kernels(p.dtype)
        if self.parameters.diagnostics_interval:
            self._diagnostic_buffer = self.xp.empty((128, self.n_bunches, 10), dtype=np.float64)
            self._diagnostic_turns = np.empty(128, dtype=np.int64)
        self._prepared = True

    def _refresh_layout(self):
        beam = self._beam
        p = beam.particles
        if p.xp is not self.xp or p.dtype != self._particle_dtype:
            raise ValueError("Feedback particle backend and precision must remain fixed after initialization")
        if self.xp is not np:
            # Raw kernels require the ParticlePool's packed, typed array contract.
            # CuPy storage/dtype are immutable; shape may change in place. Weak
            # references avoid retaining old device buffers after their replacement.
            arrays = (p.x, p.y, p.px, p.py, p.dp, p.tag, p.lost_position, p.lost_turn)
            shape = (len(p.tag), )
            if self._particle_array_refs is None or any(previous() is not value or value.shape != shape
                                                        for previous, value in zip(self._particle_array_refs, arrays)):
                for value, dtype in zip(arrays, (p.dtype, p.dtype, p.dtype, p.dtype, p.dtype, np.int32, np.float32, np.int32)):
                    if (not isinstance(value, self.xp.ndarray) or value.shape != shape or value.dtype != dtype or not value.flags.c_contiguous
                            or value.device.id != self._device_id):
                        raise ValueError("Feedback requires contiguous, equally sized particle arrays with their declared dtype and device")
                self._particle_array_refs = tuple(weakref.ref(value) for value in arrays)
        if beam.harmonic_number != self.n_bunches or len(beam.bunches) != self.n_bunches:
            raise ValueError("Transverse feedback requires its fixed injection grouping; reset/reconfigure after regrouping")
        key = (id(p), len(p.tag))
        if key == self._layout_key and p.tag is self._layout_tag:
            return
        bunches = sorted(beam.bunches, key=lambda bunch: bunch.harmonic_id)
        if [bunch.harmonic_id for bunch in bunches] != list(range(self.n_bunches)):
            raise ValueError("Feedback requires unique complete harmonic IDs")
        tiles, offsets, ranges = [], [0], []
        for slot, bunch in enumerate(bunches):
            start, end = int(bunch.start_idx), int(bunch.end_idx)
            if not 0 <= start <= end <= len(p.tag):
                raise ValueError("Invalid feedback particle range")
            ranges.append((start, end))
            for first in range(start, max(start + 1, end), 2048):
                tiles.append((slot, first, min(first + 2048, end)))
            offsets.append(len(tiles))
        expected_start = 0
        for start, end in sorted(ranges):
            if start != expected_start:
                raise ValueError("Feedback bunch ranges must partition particles without gaps or overlaps")
            expected_start = end
        if expected_start != len(p.tag):
            raise ValueError("Feedback bunch ranges must cover every particle")
        self._ranges = ranges
        self._tiles_host = np.asarray(tiles, dtype=np.int64)
        self.tiles = self.xp.asarray(self._tiles_host)
        self.offsets = self.xp.asarray(offsets, dtype=np.int64)
        self.partials = self.xp.empty((len(tiles), 2), dtype=np.float64)
        self.counts = self.xp.empty(len(tiles), dtype=np.int64)
        self._layout_key = key
        # SortBunch replaces tag together with the other particle arrays. Retain
        # the object itself so Python cannot recycle its identity before refresh.
        self._layout_tag = p.tag

    def _on_injection(self, harmonic_id, turn):
        """Reset the affected channel on every batch, including top-up injection."""
        if self._prepared and self.is_enabled and turn < self.end_turn:
            self._generation_host[harmonic_id] += 1
            self.generation[harmonic_id] = self._generation_host[harmonic_id]

    def _begin_sample(self, turn):
        if not self.is_enabled or turn >= self.end_turn:
            return False
        self._ensure_prepared()
        if turn <= self._last_pickup_turn:
            raise ValueError("Feedback pickup turns must increase strictly")
        self._refresh_layout()
        return True

    def _sample_cpu(self, turn):
        if not self._begin_sample(turn):
            return False
        p = self._beam.particles
        for slot, (start, end) in enumerate(self._ranges):
            if self.generation[slot] != self.seen_generation[slot]:
                self.history[:, slot] = 0.0
                self.pending[:, slot] = 0.0
                self.history_turn[:, slot] = -1
                self.pending_turn[:, slot] = -1
                self.seen_generation[slot] = self.generation[slot]
            row = self.latest[slot]
            row[:5], row[9] = 0.0, turn
            index = turn % self.n_taps
            target = (turn + self.delay_turns) % (self.delay_turns + 1)
            self.history[index, slot] = 0.0
            self.history_turn[index, slot] = -1
            self.pending[target, slot] = 0.0
            self.pending_turn[target, slot] = -1
            if not self._active_host[slot]:
                continue
            alive = p.tag[start:end] > 0
            n_alive = int(np.count_nonzero(alive))
            row[2] = n_alive
            if not n_alive:
                continue
            for plane, coordinate in enumerate((p.x, p.y)):
                if self.plane_mask & (1 << plane):
                    row[plane] = np.sum(coordinate[start:end][alive], dtype=np.float64) / n_alive
            if not np.all(np.isfinite(row[:2])):
                row[:2] = 0.0
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                for plane in range(2):
                    if self.plane_mask & (1 << plane):
                        self.history[index, slot, plane] = row[plane] - self.reference[slot, plane]
            if not np.all(np.isfinite(self.history[index, slot])):
                self.history[index, slot] = 0.0
                continue
            self.history_turn[index, slot] = turn
            signal = np.zeros(2, dtype=np.float64)
            valid = True
            for k in range(self.n_taps):
                previous = turn - k
                if previous < 0 or self.history_turn[previous % self.n_taps, slot] != previous:
                    valid = False
                    break
                signal += self.coefficients[k] * self.history[previous % self.n_taps, slot]
            if valid and np.all(np.isfinite(signal)):
                row[3:5] = signal
                self.pending[target, slot] = signal
                self.pending_turn[target, slot] = turn + self.delay_turns
        self._last_pickup_turn = turn
        return True

    def _compile_kernels(self, dtype):
        module = self.xp.RawModule(code=_FEEDBACK_CUDA,
                                   options=("-std=c++17", f"-DPASS_USE_FLOAT={int(dtype == np.dtype(np.float32))}", "--fmad=false"))
        self._kernels = tuple(module.get_function(name) for name in ("pickup_partial", "pickup_finalize", "feedback_kick"))

    def _sample_gpu(self, turn):
        if not self._begin_sample(turn):
            return False
        p = self._beam.particles
        partial, finish, _kick = self._kernels
        partial(((len(self._tiles_host) + 7) // 8, ), (256, ),
                (p.x, p.y, p.tag, self.tiles, self.active, self.partials, self.counts, np.int32(self.plane_mask), np.int64(len(self._tiles_host))))
        finish(((self.n_bunches + 7) // 8, ), (256, ),
               (self.partials, self.counts, self.offsets, self.active, self.generation, self.seen_generation, self.coefficients, self.reference,
                self.history, self.history_turn, self.pending, self.pending_turn, self.latest, np.int32(self.n_bunches), np.int32(
                    self.n_taps), np.int32(self.delay_turns), np.int64(turn), np.int32(self.plane_mask)))
        self._last_pickup_turn = turn
        return True

    def execute_gpu(self, sim):
        turn = int(sim.state.turn)
        if not self._begin_kick(turn):
            return False
        p = self._beam.particles
        self._kernels[2](((len(self._tiles_host) + 7) // 8, ), (256, ),
                         (p.px, p.py, p.dp, p.tag, p.lost_position, p.lost_turn, self.tiles, self.offsets, self.active, self.generation,
                          self.seen_generation, self.pending, self.pending_turn, self.gain, self.max_kick, self.latest, np.int32(self.n_bunches),
                          np.int32(self.delay_turns), np.int64(turn), np.float64(self.s), np.int32(self.plane_mask), np.int64(len(self._tiles_host))))
        self._last_kick_turn = turn
        self._record_diagnostics(turn)
        return True

    def _begin_kick(self, turn):
        if not self.is_enabled or turn < self.start_turn or turn >= self.end_turn:
            return False
        self._ensure_prepared()
        if turn <= self._last_kick_turn:
            raise ValueError("Feedback kick turns must increase strictly")
        self._refresh_layout()
        return True

    def execute_cpu(self, sim):
        turn = int(sim.state.turn)
        if not self._begin_kick(turn):
            return False
        p = self._beam.particles
        target = turn % (self.delay_turns + 1)
        for slot, (start, end) in enumerate(self._ranges):
            self.latest[slot, 5:9] = 0.0
            if not self._active_host[slot] or self.generation[slot] != self.seen_generation[slot] or self.pending_turn[target, slot] != turn:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                signal = -self.gain[slot] * self.pending[target, slot]
            if not np.all(np.isfinite(signal)):
                continue
            kick = np.clip(signal, -self.max_kick[slot], self.max_kick[slot])
            self.latest[slot, 5:7] = kick
            self.latest[slot, 7:9] = signal != kick
            if not np.any(kick):
                continue
            alive = p.tag[start:end] > 0
            px, py = p.px[start:end], p.py[start:end]
            with np.errstate(over="ignore", invalid="ignore"):
                momentum_ratio = 1.0 + p.dp[start:end].astype(np.float64)
                ps2 = momentum_ratio**2 - px.astype(np.float64)**2 - py.astype(np.float64)**2
            invalid = alive & ((momentum_ratio <= 0) | (ps2 <= 0) | ~np.isfinite(ps2))
            p.tag[start:end][invalid] = -np.abs(p.tag[start:end][invalid])
            p.lost_position[start:end][invalid] = self.s
            p.lost_turn[start:end][invalid] = turn
            alive &= ~invalid
            px[alive] += kick[0]
            py[alive] += kick[1]
            with np.errstate(over="ignore", invalid="ignore"):
                momentum_ratio = 1.0 + p.dp[start:end].astype(np.float64)
                ps2 = momentum_ratio**2 - px.astype(np.float64)**2 - py.astype(np.float64)**2
            invalid = alive & ((momentum_ratio <= 0) | (ps2 <= 0) | ~np.isfinite(ps2))
            p.tag[start:end][invalid] = -np.abs(p.tag[start:end][invalid])
            p.lost_position[start:end][invalid] = self.s
            p.lost_turn[start:end][invalid] = turn
        self._last_kick_turn = turn
        self._record_diagnostics(turn)
        return True

    def _record_diagnostics(self, turn):
        interval = self.parameters.diagnostics_interval
        if not interval or turn % interval:
            return
        self._diagnostic_buffer[self._diagnostic_size] = self.latest
        self._diagnostic_turns[self._diagnostic_size] = turn
        self._diagnostic_size += 1
        if self._diagnostic_size == len(self._diagnostic_turns):
            self._flush_diagnostics()

    def _flush_diagnostics(self):
        if not self._diagnostic_size:
            return
        if self._diagnostic_path is None:
            directory = Path(self._cfg.output_dir) / "feedback"
            directory.mkdir(parents=True, exist_ok=True)
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.cmd_name)
            suffix = hashlib.sha256(self.cmd_name.encode()).hexdigest()[:8]
            path = directory / f"beam{self.beam_id}_{name}_{suffix}.csv"
            with path.open("x", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    ("kick_turn", "harmonic_id", "centroid_x_m", "centroid_y_m", "n_alive", "filtered_x_m", "filtered_y_m", "kick_x", "kick_y",
                     "clipped_x", "clipped_y", "sample_turn", "filter_target_turn", "requested_source_turn"))
            self._diagnostic_path = path
        records = self._host(self._diagnostic_buffer[:self._diagnostic_size])
        with self._diagnostic_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for index, record in enumerate(records):
                for slot, row in enumerate(record):
                    kick_turn = int(self._diagnostic_turns[index])
                    sample_turn = int(row[9])
                    filter_target_turn = sample_turn + self.delay_turns if sample_turn >= 0 else -1
                    writer.writerow((kick_turn, slot, *row, filter_target_turn, kick_turn - self.delay_turns))
        self._diagnostic_size = 0

    def finalize(self, sim):
        self._flush_diagnostics()

    @staticmethod
    def _host(value):
        return value.get() if hasattr(value, "get") else np.asarray(value).copy()

    def _identity(self):
        return {
            "feedback": self.parameters.model_dump(mode="json"),
            "pickup": self.pickup.parameters.model_dump(mode="json"),
            "beam_id": self.beam_id,
            "n_bunches": self.n_bunches
        }

    def reset_state(self):
        if not self._prepared:
            return
        self._flush_diagnostics()
        for name in ("history", "pending", "generation", "seen_generation", "latest"):
            getattr(self, name).fill(0)
        self.latest[:, 9] = -1
        self.history_turn.fill(-1)
        self.pending_turn.fill(-1)
        self._generation_host.fill(0)
        self._last_pickup_turn = self._last_kick_turn = -1

    def state_dict(self):
        """Explicit host snapshot; restore only beside matching particles/turns."""
        if not self._prepared:
            raise RuntimeError("Prepare feedback before saving state")
        return {
            "format": "PASS-transverse-feedback-1",
            "identity": self._identity(),
            "last_pickup_turn": self._last_pickup_turn,
            "last_kick_turn": self._last_kick_turn,
            "arrays": {
                name: self._host(getattr(self, name))
                for name in ("history", "history_turn", "pending", "pending_turn", "generation", "seen_generation", "latest")
            }
        }

    def load_state_dict(self, data):
        """Validate detached state before replacing any live controller arrays."""
        if not self._prepared:
            raise RuntimeError("Prepare feedback before restoring state")
        if not isinstance(data, dict) or data.get("format") != "PASS-transverse-feedback-1" or data.get("identity") != self._identity():
            raise ValueError("Feedback checkpoint configuration does not match")
        if not isinstance(data.get("arrays"), dict):
            raise ValueError("Feedback checkpoint arrays are missing")
        for name in ("last_pickup_turn", "last_kick_turn"):
            value = data.get(name)
            if type(value) is not int or value < -1:
                raise ValueError("Invalid feedback checkpoint turn")
        host = {}
        for name in ("history", "history_turn", "pending", "pending_turn", "generation", "seen_generation", "latest"):
            template = getattr(self, name)
            if name not in data["arrays"]:
                raise ValueError(f"Feedback checkpoint array is missing: {name}")
            value = np.asarray(data["arrays"][name])
            if value.shape != template.shape or value.dtype != template.dtype:
                raise ValueError(f"Invalid feedback checkpoint shape or dtype: {name}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Nonfinite feedback checkpoint: {name}")
            host[name] = value.copy()
        for name, depth, maximum in (("history_turn", self.n_taps, data["last_pickup_turn"]), ("pending_turn", self.delay_turns + 1,
                                                                                               data["last_pickup_turn"] + self.delay_turns)):
            turns = host[name]
            valid = turns >= 0
            rows = np.broadcast_to(np.arange(depth)[:, None], turns.shape)
            if np.any(turns < -1) or np.any(turns[valid] > maximum) or np.any(turns[valid] % depth != rows[valid]):
                raise ValueError(f"Invalid feedback checkpoint ring-buffer turns: {name}")
        if np.any(host["seen_generation"] < 0) or np.any(host["generation"] < host["seen_generation"]):
            raise ValueError("Invalid feedback checkpoint injection generations")
        staged = {name: self.xp.asarray(value) for name, value in host.items()}
        self._flush_diagnostics()
        for name, value in staged.items():
            setattr(self, name, value)
        self._generation_host = self._host(self.generation)
        self._last_pickup_turn = data["last_pickup_turn"]
        self._last_kick_turn = data["last_kick_turn"]


_FEEDBACK_CUDA = r'''
#ifndef PASS_USE_FLOAT
#define PASS_USE_FLOAT 0
#endif

#if PASS_USE_FLOAT
using pass_real_t = float;
#else
using pass_real_t = double;
#endif

// All feedback kernels use 256 threads, with eight independent warp tasks.
__device__ __forceinline__ void feedback_reduce(
    double& sum_x,
    double& sum_y,
    long long& n_alive
) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_x += __shfl_down_sync(0xffffffff, sum_x, offset);
        sum_y += __shfl_down_sync(0xffffffff, sum_y, offset);
        n_alive += __shfl_down_sync(0xffffffff, n_alive, offset);
    }
}

extern "C" __global__ void pickup_partial(
    const pass_real_t* __restrict__ x,
    const pass_real_t* __restrict__ y,
    const int* __restrict__ tag,
    const long long* __restrict__ tiles,
    const int* __restrict__ active,
    double* __restrict__ partials,
    long long* __restrict__ counts,
    int plane_mask,
    long long n_tiles
) {
    const long long tile = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (tile >= n_tiles) {
        return;
    }
    const long long slot = tiles[3 * tile];
    const long long start = tiles[3 * tile + 1];
    const long long end = tiles[3 * tile + 2];
    double x_acc[4] = {0., 0., 0., 0.};
    double y_acc[4] = {0., 0., 0., 0.};
    long long n_alive = 0;
    if (active[slot]) {
        // Independent accumulators shorten rounding-error and dependency chains.
        for (long long first = start + lane; first < end; first += 128) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const long long i = first + 32 * k;
                if (i < end && tag[i] > 0) {
                    if (plane_mask & 1) {
                        x_acc[k] += (double)x[i];
                    }
                    if (plane_mask & 2) {
                        y_acc[k] += (double)y[i];
                    }
                    ++n_alive;
                }
            }
        }
    }
    double sum_x = (x_acc[0] + x_acc[1]) + (x_acc[2] + x_acc[3]);
    double sum_y = (y_acc[0] + y_acc[1]) + (y_acc[2] + y_acc[3]);
    feedback_reduce(sum_x, sum_y, n_alive);
    if (lane == 0) {
        partials[2 * tile] = sum_x;
        partials[2 * tile + 1] = sum_y;
        counts[tile] = n_alive;
    }
}

extern "C" __global__ void pickup_finalize(
    const double* __restrict__ partials,
    const long long* __restrict__ counts,
    const long long* __restrict__ offsets,
    const int* __restrict__ active,
    const long long* __restrict__ generation,
    long long* __restrict__ seen_generation,
    const double* __restrict__ coeff,
    const double* __restrict__ reference,
    double* __restrict__ history,
    long long* __restrict__ history_turn,
    double* __restrict__ pending,
    long long* __restrict__ pending_turn,
    double* __restrict__ latest,
    int n_slots,
    int n_taps,
    int delay,
    long long turn,
    int plane_mask
) {
    const long long slot = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (slot >= n_slots) {
        return;
    }
    double sum_x = 0., sum_y = 0.;
    long long n_alive = 0;
    for (long long tile = offsets[slot] + lane; tile < offsets[slot + 1]; tile += 32) {
        sum_x += partials[2 * tile];
        sum_y += partials[2 * tile + 1];
        n_alive += counts[tile];
    }
    feedback_reduce(sum_x, sum_y, n_alive);
    if (lane != 0) {
        return;
    }
    if (generation[slot] != seen_generation[slot]) {
        for (int k = 0; k < n_taps; ++k) {
            const long long row = (long long)k * n_slots + slot;
            history[2 * row] = 0.;
            history[2 * row + 1] = 0.;
            history_turn[row] = -1;
        }
        for (int k = 0; k <= delay; ++k) {
            const long long row = (long long)k * n_slots + slot;
            pending[2 * row] = 0.;
            pending[2 * row + 1] = 0.;
            pending_turn[row] = -1;
        }
        seen_generation[slot] = generation[slot];
    }

    const bool measured = active[slot] && n_alive > 0 && isfinite(sum_x) && isfinite(sum_y);
    const double centroid_x = measured && (plane_mask & 1) ? sum_x / (double)n_alive : 0.;
    const double centroid_y = measured && (plane_mask & 2) ? sum_y / (double)n_alive : 0.;
    const double signal_x = measured && (plane_mask & 1) ? centroid_x - reference[2 * slot] : 0.;
    const double signal_y = measured && (plane_mask & 2) ? centroid_y - reference[2 * slot + 1] : 0.;
    const bool valid_sample = measured && isfinite(signal_x) && isfinite(signal_y);
    const long long sample = (turn % n_taps) * n_slots + slot;
    history[2 * sample] = valid_sample ? signal_x : 0.;
    history[2 * sample + 1] = valid_sample ? signal_y : 0.;
    history_turn[sample] = valid_sample ? turn : -1;

    bool ready = valid_sample && turn >= n_taps - 1;
    double filtered_x = 0., filtered_y = 0.;
    if (ready) {
        for (int k = 0; k < n_taps; ++k) {
            const long long source_turn = turn - k;
            const long long row = (source_turn % n_taps) * n_slots + slot;
            if (history_turn[row] != source_turn) {
                ready = false;
                break;
            }
            filtered_x += coeff[2 * k] * history[2 * row];
            filtered_y += coeff[2 * k + 1] * history[2 * row + 1];
        }
    }
    ready = ready && isfinite(filtered_x) && isfinite(filtered_y);
    if (!ready) {
        filtered_x = 0.;
        filtered_y = 0.;
    }
    // The extra queue slot prevents overwriting the current turn before its kick.
    const long long target_turn = turn + delay;
    const long long target = (target_turn % (delay + 1)) * n_slots + slot;
    pending[2 * target] = filtered_x;
    pending[2 * target + 1] = filtered_y;
    pending_turn[target] = ready ? target_turn : -1;

    latest[10 * slot] = centroid_x;
    latest[10 * slot + 1] = centroid_y;
    latest[10 * slot + 2] = (double)n_alive;
    latest[10 * slot + 3] = filtered_x;
    latest[10 * slot + 4] = filtered_y;
    latest[10 * slot + 9] = (double)turn;
}

extern "C" __global__ void feedback_kick(
    pass_real_t* __restrict__ px,
    pass_real_t* __restrict__ py,
    const pass_real_t* __restrict__ dp,
    int* __restrict__ tag,
    float* __restrict__ lost_position,
    int* __restrict__ lost_turn,
    const long long* __restrict__ tiles,
    const long long* __restrict__ offsets,
    const int* __restrict__ active,
    const long long* __restrict__ generation,
    const long long* __restrict__ seen_generation,
    const double* __restrict__ pending,
    const long long* __restrict__ pending_turn,
    const double* __restrict__ gain,
    const double* __restrict__ max_kick,
    double* __restrict__ latest,
    int n_slots,
    int delay,
    long long turn,
    double s,
    int plane_mask,
    long long n_tiles
) {
    const long long tile = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (tile >= n_tiles) {
        return;
    }
    const long long slot = tiles[3 * tile];
    const long long start = tiles[3 * tile + 1];
    const long long end = tiles[3 * tile + 2];
    double kick_x = 0., kick_y = 0.;
    if (lane == 0) {
        const long long row = (turn % (delay + 1)) * n_slots + slot;
        const bool ready = active[slot] && generation[slot] == seen_generation[slot] && pending_turn[row] == turn;
        double raw_x = 0., raw_y = 0.;
        if (ready) {
            if (plane_mask & 1) {
                raw_x = -gain[2 * slot] * pending[2 * row];
            }
            if (plane_mask & 2) {
                raw_y = -gain[2 * slot + 1] * pending[2 * row + 1];
            }
        }
        if (!isfinite(raw_x) || !isfinite(raw_y)) {
            raw_x = 0.;
            raw_y = 0.;
        }
        kick_x = fmin(fmax(raw_x, -max_kick[2 * slot]), max_kick[2 * slot]);
        kick_y = fmin(fmax(raw_y, -max_kick[2 * slot + 1]), max_kick[2 * slot + 1]);
        if (tile == offsets[slot]) {
            latest[10 * slot + 5] = kick_x;
            latest[10 * slot + 6] = kick_y;
            latest[10 * slot + 7] = raw_x != kick_x;
            latest[10 * slot + 8] = raw_y != kick_y;
        }
    }
    kick_x = __shfl_sync(0xffffffff, kick_x, 0);
    kick_y = __shfl_sync(0xffffffff, kick_y, 0);
    if (kick_x == 0. && kick_y == 0.) {
        return;
    }
    for (long long i = start + lane; i < end; i += 32) {
        if (tag[i] <= 0) {
            continue;
        }
        const pass_real_t dp_i = dp[i];
        const double px_old = (double)px[i], py_old = (double)py[i];
        const pass_real_t px_new = (pass_real_t)(px_old + kick_x);
        const pass_real_t py_new = (pass_real_t)(py_old + kick_y);
        // Here P/P0 >= 0.5 and px^2+py^2 <= 0.125, so ps^2 >= 0.125.
        // Check both states before writing; nonfinite values take the full path.
        const bool safe = fabs(dp_i) <= (pass_real_t)0.5 && fabs(px_old) <= 0.25 && fabs(py_old) <= 0.25 && fabs(px_new) <= (pass_real_t)0.25 &&
                          fabs(py_new) <= (pass_real_t)0.25;
        double momentum_ratio = 0.;
        if (!safe) {
            momentum_ratio = 1. + (double)dp_i;
            const double ps2 = momentum_ratio * momentum_ratio - px_old * px_old - py_old * py_old;
            if (!(momentum_ratio > 0.) || !(ps2 > 0.) || !isfinite(ps2)) {
                tag[i] = -tag[i];
                lost_position[i] = (float)s;
                lost_turn[i] = (int)turn;
                continue;
            }
        }
        if (plane_mask & 1) {
            px[i] = px_new;
        }
        if (plane_mask & 2) {
            py[i] = py_new;
        }
        if (!safe) {
            const double ps2 = momentum_ratio * momentum_ratio - (double)px_new * px_new - (double)py_new * py_new;
            if (!(ps2 > 0.) || !isfinite(ps2)) {
                tag[i] = -tag[i];
                lost_position[i] = (float)s;
                lost_turn[i] = (int)turn;
            }
        }
    }
}
'''
