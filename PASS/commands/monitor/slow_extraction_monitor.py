"""Read-only, cumulative spill histograms from SlowExtraction event batches."""

from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np

from PASS.commands.command import Command
from PASS.para.schema.slow_extraction import SlowExtractionMonitorItem
from PASS.utils.constants import const
from PASS.utils.table_io import append_table, write_table


@Command.register("slowextractionmonitor")
class SlowExtractionMonitor(Command):
    """Accumulate source events without reading or changing particle storage.

    Turn and physical-time gates intersect. Turn bins start at Start turn;
    time bins start at Time origin and use the captured particle arrival times.
    Nominal bin edges remain fixed between snapshots. The separate observed
    edges clip the source-reference/event envelope to the statistical gates.
    Rates use positive observed widths and are NaN for zero-width coverage.
    Partial bins must not be compared directly with full-bin spill metrics.
    All time bins remain provisional: later source batches can contain an
    earlier arrival time, so no bin is closed while tracking continues.
    Completed turn bins are appended once; the last partial turn bin is
    appended at finalization. Time output remains an atomic full snapshot.
    """

    def __init__(self, beam_id, sim, **command_kwargs):
        kwargs = {str(k).lower(): v for k, v in command_kwargs.items()}
        self.cmd_name = str(kwargs.pop("name"))
        aliases = {}
        for name, field in SlowExtractionMonitorItem.model_fields.items():
            aliases[name.lower()] = field.alias or name
            aliases[(field.alias or name).lower()] = field.alias or name
        settings = SlowExtractionMonitorItem.model_validate({aliases.get(k, k): v for k, v in kwargs.items()})
        for name in SlowExtractionMonitorItem.model_fields:
            setattr(self, name, getattr(settings, name))
        self.beam_id = int(beam_id)
        self.cmd_type = type(self).__name__
        self._output_dir = Path(sim.cfg.output_dir_stat)
        if not getattr(sim.cfg, "flat_output", False):
            self._output_dir /= "slow_extraction"
        self.output_paths = {}
        self._source = None
        self._last_serial = None
        self._histograms = {kind: {} for kind in (("turn", "time") if self.bin_by == "both" else (self.bin_by, ))}
        self._turn_range = None
        self._reference_range = None
        self._event_time_range = None
        self._created_paths = set()
        self._next_unwritten_turn_bin = None
        self._written_turn_count = 0
        self._written_turn_real = 0.
        self._output_failed = False
        self._dirty = False
        self._finalized = False
        super().__init__()

    def print(self):
        logging.getLogger(__name__).info("S=%g, Command=%s, Name=%s, Source=%s, BinBy=%s", self.s, self.cmd_type, self.cmd_name, self.source,
                                         self.bin_by)

    def _resolve_source(self, sim):
        source = getattr(sim, "_slow_extraction_sources", {}).get((self.beam_id, self.source.lower()))
        if source is None:
            raise ValueError(f"SlowExtractionMonitor '{self.cmd_name}': unknown Source '{self.source}'")
        if not math.isclose(float(source.s), self.s, rel_tol=0., abs_tol=const.eps):
            raise ValueError(f"SlowExtractionMonitor '{self.cmd_name}' and Source '{self.source}' must be at the same S (m)")
        if self._source is not None and source is not self._source:
            raise ValueError(f"SlowExtractionMonitor '{self.cmd_name}': Source changed during tracking")
        if self._source is None:
            self._source = source
            label = f"{source.cmd_name}_{self.cmd_name}"
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "monitor"
            digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:8]
            run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source.run_id))
            stem = f"{slug}_{digest}_{run_id}_beam{self.beam_id}"
            self.output_paths = {kind: self._output_dir / f"{stem}_{kind}.h5" for kind in self._histograms}
        return source

    @staticmethod
    def _extend_range(current, lower, upper):
        if current is None:
            return (lower, upper)
        return (min(current[0], lower), max(current[1], upper))

    def execute_cpu(self, sim):
        return self._execute(sim)

    def execute_gpu(self, sim):
        return self._execute(sim)

    def _execute(self, sim):
        if self._finalized:
            raise RuntimeError(f"SlowExtractionMonitor '{self.cmd_name}' cannot execute after finalization")
        if self._output_failed:
            raise RuntimeError(f"SlowExtractionMonitor '{self.cmd_name}' cannot continue after an output failure")
        source = self._resolve_source(sim)
        turn = int(sim.state.turn)
        if source.last_turn != turn:
            raise ValueError(f"SlowExtractionMonitor '{self.cmd_name}' must execute after Source '{self.source}' on every turn")
        serial = int(source.batch_serial)
        if serial == self._last_serial:
            return False
        if self._last_serial is not None and serial != self._last_serial + 1:
            raise ValueError(f"SlowExtractionMonitor '{self.cmd_name}': Source batch serial is not consecutive; events may have been missed")
        self._last_serial = serial
        if turn < self.start_turn or (self.end_turn is not None and turn >= self.end_turn):
            return False

        self._turn_range = self._extend_range(self._turn_range, turn, turn + 1)
        reference_times = np.asarray(source.reference_times, dtype=np.float64)
        if reference_times.size:
            if not np.all(np.isfinite(reference_times)):
                raise ValueError("SlowExtraction source reference times must be finite")
            self._reference_range = self._extend_range(self._reference_range, float(reference_times.min()), float(reference_times.max()))

        events = source.last_events
        times = np.asarray(events["time"], dtype=np.float64)
        turns = np.asarray(events["turn"], dtype=np.int64)
        selected = (turns >= self.start_turn)
        if self.end_turn is not None:
            selected &= turns < self.end_turn
        if self.start_time is not None:
            selected &= times >= self.start_time
        if self.end_time is not None:
            selected &= times < self.end_time
        times, turns = times[selected], turns[selected]
        if len(times):
            if not np.all(np.isfinite(times)):
                raise ValueError("SlowExtraction event times must be finite")
            weights = np.asarray(events["macro_weight"], dtype=np.float64)[selected]
            charges = np.asarray(events["charge_number"], dtype=np.float64)[selected] * const.e * weights
            self._event_time_range = self._extend_range(self._event_time_range, float(times.min()), float(times.max()))
            for kind, histogram in self._histograms.items():
                if kind == "turn":
                    indices = (turns - self.start_turn) // self.turn_bin_width
                else:
                    with np.errstate(over="ignore", invalid="ignore"):
                        raw_indices = np.floor((times - self.time_origin) / self.time_bin_width)
                    if np.any(~np.isfinite(raw_indices)) or np.any(raw_indices >= float(2**63)) or np.any(raw_indices < -float(2**63)):
                        raise ValueError("SlowExtractionMonitor time-bin indices must fit int64; increase Time bin width (s)")
                    indices = raw_indices.astype(np.int64)
                unique, inverse = np.unique(indices, return_inverse=True)
                counts = np.bincount(inverse)
                real = np.bincount(inverse, weights=weights)
                charge = np.bincount(inverse, weights=charges)
                for index, n_particles, n_real, extracted_charge in zip(unique, counts, real, charge):
                    previous = histogram.get(int(index), (0, 0., 0.))
                    histogram[int(index)] = (previous[0] + int(n_particles), previous[1] + float(n_real), previous[2] + float(extracted_charge))
        self._dirty = True
        if (turn + 1) % self.write_interval_turns == 0 or turn == int(sim.cfg.num_turn) - 1:
            self._write_snapshots()
        return True

    def _observed_range(self, kind):
        if kind == "turn":
            return self._turn_range
        observed = self._reference_range
        if self._event_time_range is not None:
            observed = self._extend_range(observed, *self._event_time_range)
        if observed is None:
            return None
        lower, upper = observed
        if self.start_time is not None:
            lower = max(lower, self.start_time)
        if self.end_time is not None:
            upper = min(upper, self.end_time)
        if upper < lower or (self.end_time is not None and lower >= self.end_time):
            return None
        return lower, upper

    def get_histogram(self, bin_by):
        """Return an owned host table, including observed zero-count bins.

        A last event exactly on a bin boundary retains its zero-width endpoint
        bin until later observations give it coverage. The bin's rates are NaN.
        """
        return self._build_histogram(bin_by)

    def _build_histogram(self, bin_by, start_index=None, stop_index=None, cumulative_count=0, cumulative_real=0.):
        """Build only the requested rows; turn appends never scan old bins."""
        if bin_by not in self._histograms:
            raise ValueError(f"SlowExtractionMonitor '{self.cmd_name}' does not collect '{bin_by}' bins")
        histogram = self._histograms[bin_by]
        observed = self._observed_range(bin_by)
        origin = self.start_turn if bin_by == "turn" else self.time_origin
        width = self.turn_bin_width if bin_by == "turn" else self.time_bin_width
        if observed is None:
            indices = np.empty(0, dtype=np.int64)
        else:
            lower, upper = observed
            first_position, end_position = (lower - origin) / width, (upper - origin) / width
            if not all(math.isfinite(position) and -float(2**63) <= position < float(2**63) for position in (first_position, end_position)):
                raise ValueError("SlowExtractionMonitor observed bin indices must fit int64; increase the bin width")
            first = math.floor(first_position)
            last = max(first, math.ceil(end_position) - 1)
            if bin_by == "time" and histogram:
                first, last = min(first, min(histogram)), max(last, max(histogram))
            if start_index is not None:
                first = max(first, start_index)
            if stop_index is not None:
                last = min(last, stop_index - 1)
            indices = np.arange(first, last + 1, dtype=np.int64)
        nominal_start = origin + indices * width
        nominal_end = nominal_start + width
        observed_start = np.maximum(nominal_start, observed[0]) if observed is not None else nominal_start.copy()
        observed_end = np.minimum(nominal_end, observed[1]) if observed is not None else nominal_end.copy()
        observed_width = np.maximum(observed_end - observed_start, 0.)
        counts = np.zeros(len(indices), dtype=np.int64)
        real = np.zeros(len(indices), dtype=np.float64)
        charge = np.zeros(len(indices), dtype=np.float64)
        for row, index in enumerate(indices):
            if int(index) in histogram:
                counts[row], real[row], charge[row] = histogram[int(index)]
        table = {
            "bin_start": nominal_start,
            "bin_end": nominal_end,
            "observed_start": observed_start,
            "observed_end": observed_end,
            "observed_width": observed_width,
            "is_partial": (observed_start > nominal_start) | (observed_end < nominal_end),
            "num_extracted": counts,
            "real_extracted": real,
            "charge_extracted": charge,
            "cumulative_extracted": cumulative_count + np.cumsum(counts),
            "cumulative_real": cumulative_real + np.cumsum(real),
        }
        if bin_by == "time":
            table["particle_rate"] = np.divide(real, observed_width, out=np.full(len(indices), np.nan), where=observed_width > 0.)
            table["current"] = np.divide(charge, observed_width, out=np.full(len(indices), np.nan), where=observed_width > 0.)
        return table

    def _headers(self, kind):
        source = self._source
        reference = self._reference_range or (np.nan, np.nan)
        events = self._event_time_range or (np.nan, np.nan)
        output_policy = ("completed turn bins appended once; final partial bin included at finalization"
                         if kind == "turn" else "atomic provisional snapshot; earlier time bins may change")
        return {
            "Name": "PASS Slow Extraction Spill",
            "Monitor": self.cmd_name,
            "Source": source.cmd_name,
            "SourceRunId": str(source.run_id),
            "SourceEvents": str(source.output_path),
            "BeamId": self.beam_id,
            "S": self.s,
            "BinBy": kind,
            "BinInterval": "[bin_start, bin_end)",
            "TimeDefinition": "time = reference_time - z / (reference_beta*c); no folding",
            "ReferenceObservedStart": reference[0],
            "ReferenceObservedEnd": reference[1],
            "EventObservedStart": events[0],
            "EventObservedEnd": events[1],
            "StartTurn": self.start_turn,
            "EndTurn": self.end_turn if self.end_turn is not None else -1,
            "StartTime": self.start_time if self.start_time is not None else np.nan,
            "EndTime": self.end_time if self.end_time is not None else np.nan,
            "TimeOrigin": self.time_origin,
            "Coverage": "source reference/event envelope clipped to gates; partial bins are provisional",
            "RateDefinition": "real_extracted/observed_width; signed current=charge_extracted/observed_width; NaN for zero width",
            "CountDefinition": "source events passing both turn and physical-time gates",
            "OutputPolicy": output_policy,
        }

    def _write_turn_bins(self, final):
        path = self.output_paths["turn"]
        if self._turn_range is None:
            if final and path not in self._created_paths:
                self._write_atomic_snapshot("turn")
            return
        lower, upper = self._turn_range
        if self._next_unwritten_turn_bin is None:
            self._next_unwritten_turn_bin = (lower - self.start_turn) // self.turn_bin_width
        stop = (upper - self.start_turn) // self.turn_bin_width
        if final and (upper - self.start_turn) % self.turn_bin_width:
            stop += 1
        if stop <= self._next_unwritten_turn_bin:
            return
        if path not in self._created_paths and path.exists():
            raise FileExistsError(f"SlowExtractionMonitor refuses to overwrite existing output: {path}")
        headers = self._headers("turn")
        while self._next_unwritten_turn_bin < stop:
            end = min(stop, self._next_unwritten_turn_bin + 65536)
            table = self._build_histogram("turn", self._next_unwritten_turn_bin, end, self._written_turn_count, self._written_turn_real)
            append_table(path,
                         table,
                         headers,
                         chunk_rows=min(65536, max(1, self.write_interval_turns // self.turn_bin_width)),
                         output_format=self.output_format)
            self._created_paths.add(path)
            self._next_unwritten_turn_bin = end
            self._written_turn_count = int(table["cumulative_extracted"][-1])
            self._written_turn_real = float(table["cumulative_real"][-1])
        with h5py.File(path, "a") as stream:
            for name, value in headers.items():
                stream.attrs[name] = value

    def _write_atomic_snapshot(self, kind):
        path = self.output_paths[kind]
        if path not in self._created_paths and path.exists():
            raise FileExistsError(f"SlowExtractionMonitor refuses to overwrite existing output: {path}")
        with tempfile.NamedTemporaryFile(prefix=path.stem + "_", suffix=".h5", dir=self._output_dir, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            write_table(temporary, self.get_histogram(kind), self._headers(kind), output_format=self.output_format)
            if path in self._created_paths:
                os.replace(temporary, path)
            else:
                # Linking publishes the first complete snapshot without
                # replacing a pre-existing path, even in a naming race.
                os.link(temporary, path)
                self._created_paths.add(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _write_snapshots(self, final=False):
        if self._output_failed or (not self._dirty and not final):
            return
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            if "turn" in self.output_paths:
                self._write_turn_bins(final)
            if "time" in self.output_paths and (self._dirty or self.output_paths["time"] not in self._created_paths):
                self._write_atomic_snapshot("time")
            self._dirty = False
        except BaseException:
            self._output_failed = True
            raise

    def finalize(self, sim):
        """Publish a last, possibly empty snapshot; never retry a failed write."""
        if self._finalized:
            return
        if self._output_failed:
            self._finalized = True
            return
        self._resolve_source(sim)
        if not self._created_paths:
            self._dirty = True
        self._write_snapshots(final=True)
        self._finalized = True
