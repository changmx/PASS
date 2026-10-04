"""Bounded reads of the six coordinates used by file-based Injection."""

from pathlib import Path
import shlex

import h5py
import numpy as np


class DistributionBatchReader:
    """Read distribution rows without retaining coordinates or live file handles.

    TFS byte offsets are cached at requested batch boundaries. Shallow copies of
    an injection source can share this reader, including after a failed batch.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._fields = ("x", "px", "y", "py", "z", "dp")
        self._tfs_identity = None
        self._tfs_positions = {}
        self._column_indices = None

    def read_rows(self, start, count):
        """Return only rows [start, start+count), in x,px,y,py,z,dp order."""
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or start < 0:
            raise ValueError("Distribution row start must be a nonnegative integer")
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Distribution row count must be a nonnegative integer")
        start, count = int(start), int(count)
        if self.path.suffix.lower() in (".h5", ".hdf5"):
            return self._read_hdf5_rows(start, count)
        return self._read_tfs_rows(start, count)

    def _read_hdf5_rows(self, start, count):
        end = start + count
        with h5py.File(self.path, "r") as stream:
            lengths = set()
            for name in self._fields:
                if name not in stream:
                    raise ValueError(f"Distribution {self.path} is missing coordinate {name!r}")
                dataset = stream[name]
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1 or dataset.dtype.kind not in "biuf":
                    raise ValueError(f"Distribution {self.path} requires one-dimensional numeric coordinate {name!r}")
                lengths.add(dataset.shape[0])
            if len(lengths) != 1:
                raise ValueError(f"Distribution {self.path} coordinate columns have unequal lengths")
            n_rows = lengths.pop()
            if end > n_rows:
                raise ValueError(f"Distribution {self.path} needs {end} rows; found {n_rows}")
            values = np.empty((count, len(self._fields)), dtype=np.float64)
            for column, name in enumerate(self._fields):
                values[:, column] = stream[name][start:end]
        return values

    def _read_tfs_header(self, stream):
        columns = None
        while raw := stream.readline():
            line = raw.decode("utf-8-sig").strip()
            if line.startswith("*"):
                columns = shlex.split(line[1:])
            elif line.startswith("$"):
                if columns is None or len(shlex.split(line[1:])) != len(columns):
                    raise ValueError(f"Distribution {self.path} has invalid TFS column/type declarations")
                if len(set(columns)) != len(columns):
                    raise ValueError(f"Distribution {self.path} has duplicate TFS columns")
                missing = [name for name in self._fields if name not in columns]
                if missing:
                    raise ValueError(f"Distribution {self.path} is missing coordinates {missing}")
                self._column_indices = tuple(columns.index(name) for name in self._fields)
                self._tfs_positions = {0: stream.tell()}
                return
            elif line and not line.startswith(("@", "#", "!")):
                raise ValueError(f"Distribution {self.path} has no TFS column/type declarations before its data")
        raise ValueError(f"Distribution {self.path} has no complete TFS header")

    @staticmethod
    def _read_data_line(stream):
        while raw := stream.readline():
            line = raw.strip()
            if line and not line.startswith((b"#", b"!", b"@")):
                return raw
        return None

    def _read_tfs_rows(self, start, count):
        metadata = self.path.stat()
        identity = (metadata.st_size, metadata.st_mtime_ns)
        if identity != self._tfs_identity:
            self._tfs_identity = None
            self._tfs_positions = {}
            self._column_indices = None
        end = start + count
        with self.path.open("rb") as stream:
            if self._column_indices is None:
                self._read_tfs_header(stream)
                self._tfs_identity = identity
            row = start if start in self._tfs_positions else max(position for position in self._tfs_positions if position <= start)
            stream.seek(self._tfs_positions[row])
            # A fresh reader can resume at an arbitrary row without parsing
            # preceding values. Normal sequential batches seek directly.
            while row < start:
                if self._read_data_line(stream) is None:
                    raise ValueError(f"Distribution {self.path} needs {end} rows; found {row}")
                row += 1
            self._tfs_positions[start] = stream.tell()
            lines = []
            while row < end:
                raw = self._read_data_line(stream)
                if raw is None:
                    raise ValueError(f"Distribution {self.path} needs {end} rows; found {row}")
                lines.append(raw)
                row += 1
            next_offset = stream.tell()
        if not count:
            return np.empty((0, len(self._fields)), dtype=np.float64)
        try:
            values = np.loadtxt(lines, dtype=np.float64, usecols=self._column_indices, ndmin=2, encoding="utf-8", quotechar='"')
        except ValueError as exc:
            raise ValueError(f"Distribution {self.path} has invalid coordinates in rows [{start}, {end})") from exc
        self._tfs_positions[end] = next_offset
        return values
