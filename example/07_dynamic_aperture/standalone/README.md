# Portable dynamic-aperture analysis

Copy **`dynamic_aperture.py`** and your ParticleMonitor file to another directory
or computer. The analysis script does not require PASS, its GUI, or the original
simulation input. It reads the current single-file ParticleMonitor format
(version 2), including the captured injection coordinates used for x, y, and dp
grouping. Older monitor formats are unsupported.

Use Python 3.11 or newer and install the three runtime dependencies:

```console
python -m pip install numpy h5py matplotlib
```

## Command line

Overlay the sampled boundaries of every initial dp group:

```console
python dynamic_aperture.py ParticleMonitor.h5 --overlay --output aperture.png
```

Plot one group at an explicitly selected monitor turn, and save the analysis data:

```console
python dynamic_aperture.py ParticleMonitor.h5 --turn 1023 --dp 0 --output aperture.pdf --data aperture.npz
```

Select a subset for an overlay:

```console
python dynamic_aperture.py ParticleMonitor.h5 --overlay --dp-values -0.01 0 0.01 --output aperture.svg
```

Use `--mode loss_turn` for a single-dp scatter plot colored by the absolute loss
turn. Without `--dp` or `--overlay`, the first sorted initial dp group is plotted.
The requested output directory must already exist. The command refuses to
overwrite any existing figure, data file, or CSV metadata sidecar.

`--turn` is an inclusive **simulation turn label at the ParticleMonitor**, not a
number of completed revolutions or the age of an injected particle. If omitted,
the reader uses the monitor's `RequestedEndTurn - 1` (falling back to `EndTurn - 1`
or the last recorded sample). A live particle without a sample on the requested
turn is marked incomplete; an earlier recorded loss remains a known loss. For a
particular horizon, configure the monitor to record that turn.

The GUI's **End turn** field clamps a manually entered turn beyond the last
committed sample and displays the adjustment. The Python reader and CLI retain
strict requested-turn behavior by default. To opt in from Python, pass
`clamp_to_available=True` to `read_dynamic_aperture`. Result metadata preserves
`requested_turn_input`, the effective `requested_turn`, and `last_sample_turn`.
Automatic selection, gaps within the recorded range, and empty histories are
not clamped. In particular, an interrupted run is not silently treated as
complete just because automatic mode was selected.

## Call from Python

Place this file alongside your own script or on its Python import path:

```python
import matplotlib.pyplot as plt
from dynamic_aperture import (
    read_dynamic_aperture,
    plot_dynamic_aperture,
    plot_dynamic_aperture_boundaries,
    export_dynamic_aperture,
)

result = read_dynamic_aperture("ParticleMonitor.h5", requested_turn=1023)

ax = plot_dynamic_aperture(result, dp=0.0)
ax.figure.savefig("single_dp.png", dpi=200)

fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
plot_dynamic_aperture_boundaries(result, ax=ax)
fig.savefig("dp_overlay.pdf")

export_dynamic_aperture(result, "aperture.npz")
plt.show()
```

Importing the module does not change Matplotlib's backend. Its command-line
entry point uses the noninteractive Agg backend for saving figures. Plotting
functions return Matplotlib `Axes`, so callers can adjust titles, limits, legends,
and figure size. The Python functions follow the PASS API: `savefig` and
`export_dynamic_aperture` can overwrite their requested paths; the command-line
overwrite guard is not an API restriction.

Numeric exports are staged before publication. A caught publication failure
restores prior CSV/JSON files; if restoration also fails, the exception names
the retained recovery directory. This does not provide a multi-file transaction
across process or machine crashes. Uppercase `.NPZ` preserves the exact requested
filename. The CLI removes only files it created if output publication fails.

## What is saved and what a boundary means

- PNG, PDF, and SVG contain the Matplotlib figure, including its legend and
  annotations. They are not Python source files.
- NPZ contains the per-particle analysis arrays, including initial and selected
  final coordinates, statuses, loss information, and JSON metadata. Load it with
  `numpy.load("aperture.npz", allow_pickle=False)`.
- CSV contains one analysis row per particle, including initial coordinates,
  status, sampled turn, and loss information. A neighboring JSON file contains
  metadata. These are analysis summaries, not full turn-by-turn trajectories.
- Data exports include **all** dp groups in the loaded result, even when the
  figure selects one group or a subset. Original x, y, and z are in metres;
  displayed transverse positions are in millimetres.
- Boundaries interpolate sampled survival/loss transitions at fixed initial
  px, py, and z within each dp group. They retain holes and disconnected islands,
  mask unknown states, and never close a boundary along the edge of the scan.
  An upper-half-plane grid is supported without mirroring the unscanned half.
  A boundary is a finite-turn, finite-grid estimate rather than a proof of
  indefinite stability.

HDF5 is the preferred input for speed: analysis reads the initial coordinates and
one selected trajectory sample rather than the full history. TFS input is read
in bounded row blocks but must scan the complete table to validate it. Neither
reader requires importing the full PASS package.

## Updating the portable file inside PASS

This file is generated from the same reader, classification, and plotting source
used by PASS. Do not maintain a separate copy of the algorithms by hand. From
the repository root, run:

```console
python tools/export_dynamic_aperture.py
python tools/export_dynamic_aperture.py --check
```

The exporter itself uses only Python's standard library. Source hashes in the
generated module record its provenance. Regenerate and recopy the portable file
after changing the maintained implementation; an already copied file does not
update automatically.
