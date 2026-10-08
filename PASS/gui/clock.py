"""Inspect the automatically derived clock without allocating particles."""

from PASS.utils.reference_clock import build_reference_program


def reference_clock_snapshot(data, base_dir=None):
    program = build_reference_program(data, base_dir)
    scalar = len(program.values) == 1
    return {
        "Time origin (s)": program.origin,
        "Revolution frequency (Hz)": float(program.values[0]) if scalar else program.values.tolist(),
        "Time (s)": None if scalar else program.times.tolist(),
    }
