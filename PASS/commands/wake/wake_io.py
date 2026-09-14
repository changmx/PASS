"""Explicit, audited numeric table conversion into canonical integrated SI data."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WakeConvention:
    data_kind: Literal["wake_function", "wake_potential", "impedance"]
    axis: Literal["time", "distance", "frequency"]
    axis_unit: str
    value_unit: str
    positive_trailing: bool
    longitudinal_positive_loss: bool
    fourier_exponent: Literal[-1, 1]
    transverse_impedance_factor: Literal["i", "-i", "1"]
    shunt_impedance_convention: str
    integrated: bool
    reference_beta: float

    def __post_init__(self):
        import math
        if self.data_kind not in {"wake_function", "wake_potential", "impedance"} or self.axis not in {"time", "distance", "frequency"}:
            raise ValueError("Invalid wake file data kind or axis")
        for name in ("positive_trailing", "longitudinal_positive_loss", "integrated"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"Wake convention {name} must be a boolean")
        if isinstance(self.fourier_exponent, bool) or self.fourier_exponent not in {-1, 1}:
            raise ValueError("Fourier exponent must be -1 or 1")
        if self.transverse_impedance_factor not in {"i", "-i", "1"}:
            raise ValueError("Transverse impedance factor must be i, -i or 1")
        if isinstance(self.reference_beta, bool) or not math.isfinite(self.reference_beta) or not 0 < self.reference_beta <= 1:
            raise ValueError("File reference beta must lie in (0, 1]")


def _value_scale(unit, kind, order):
    """Accept a small explicit unit grammar; never guess dimensional powers."""
    import re
    unit = unit.replace(" ", "").replace("(", "").replace(")", "").replace(".", "*")
    unit = unit.replace("**", "^").replace("*", "/")
    parts = unit.split("/")
    if kind == "wake_function":
        charges = [part for part in parts[1:] if part in {"C", "nC", "pC"}]
        if len(parts) < 2 or parts[0] not in {"V", "kV", "MV"} or len(charges) != 1:
            raise ValueError("Wake units must explicitly specify voltage / charge / spatial powers")
        scale = {"V": 1., "kV": 1e3, "MV": 1e6}[parts.pop(0)]
        scale /= {"C": 1., "nC": 1e-9, "pC": 1e-12}[charges[0]]
        parts.remove(charges[0])
    else:
        if parts[0] not in {"ohm", "Ohm", "kOhm", "MOhm"}:
            raise ValueError("Impedance units must specify ohm / spatial powers")
        scale = {"ohm": 1., "Ohm": 1., "kOhm": 1e3, "MOhm": 1e6}[parts.pop(0)]
    degree = 0
    for part in parts:
        match = re.fullmatch(r"(m|cm|mm)(?:\^(\d+))?", part)
        if match is None:
            raise ValueError(f"Unsupported wake unit term {part!r}")
        power = int(match[2] or 1)
        degree += power
        scale /= {"m": 1., "cm": .01, "mm": .001}[match[1]]**power
    if degree != order:
        raise ValueError(f"Wake unit spatial order {degree} does not match required order {order}")
    return scale


def read_wake_file(path, *, convention: WakeConvention, component: str, spatial=None,
                   format="table", axis_column=0, value_column=1, imag_column=None,
                   delimiter=None, skiprows=0, causal=True, reconstruction="two_sided", length=None):
    """Read selected zero-based columns of a whitespace/CSV numeric table.

    HEADTAIL is an explicit ns, V/pC, V/(pC*mm) format contract, with
    user-selected columns because several HEADTAIL table layouts exist.
    A wake potential is not a point-charge wake and needs separate deconvolution.
    """
    import hashlib
    import io
    from pathlib import Path
    from dataclasses import asdict
    import numpy as np
    from PASS.utils.constants import const
    from .wake_components import COMPONENTS, SpatialTerm
    from .wake_models import TabulatedWakeModel
    from .wake_spectrum import ImpedanceSpectrum, SpectrumWakeModel

    c = convention
    if not isinstance(c, WakeConvention):
        raise TypeError("Supply an explicit WakeConvention for file units, signs and normalization")
    if c.data_kind not in {"wake_function", "impedance"}:
        raise ValueError("Wake-potential input requires explicit deconvolution; supply a wake function or impedance")
    if not np.isfinite(c.reference_beta) or not 0 < c.reference_beta <= 1:
        raise ValueError("File reference beta must lie in (0, 1]")
    if c.fourier_exponent not in {-1, 1} or c.transverse_impedance_factor not in {"i", "-i", "1"}:
        raise ValueError("Invalid Fourier convention")
    if (component == "custom") != (spatial is not None):
        raise ValueError("Only custom components require explicit spatial powers")
    term = spatial if spatial is not None else SpatialTerm(*COMPONENTS[component])
    order = sum(term.source_powers)+sum(term.test_powers)
    scale = _value_scale(c.value_unit, c.data_kind, order+int(not c.integrated))
    if not c.integrated:
        if length is None or not np.isfinite(length) or length <= 0:
            raise ValueError("A per-length wake/impedance requires positive physical Length (m)")
        scale *= length
    elif length is not None:
        raise ValueError("Integrated wake data must not be multiplied by another length")
    if format not in {"table", "headtail"}:
        raise ValueError("Supported numeric file formats are table and headtail")
    if format == "headtail":
        if (c.data_kind != "wake_function" or c.axis != "time" or c.axis_unit != "ns"
                or not c.integrated or order > 1
                or _value_scale(c.value_unit, c.data_kind, order) != 1e12*1e3**order):
            raise ValueError("HEADTAIL requires ns and integrated V/pC or V/(pC*mm) units")
    columns = [axis_column, value_column]+([] if imag_column is None else [imag_column])
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in columns+[skiprows]) or len(set(columns)) != len(columns):
        raise ValueError("File columns must be distinct nonnegative integers; Skip rows >= 0")
    payload = Path(path).read_bytes()
    table = np.loadtxt(io.StringIO(payload.decode("utf-8-sig")), delimiter=delimiter, skiprows=skiprows, ndmin=2)
    if len(table) < 2 or max(columns) >= table.shape[1] or not np.all(np.isfinite(table[:, columns])):
        raise ValueError("Wake file needs at least two finite rows and the declared columns")
    axis = table[:, axis_column].copy()
    values = table[:, value_column]*scale
    if term.plane == "z" and not c.longitudinal_positive_loss:
        values = -values
    if c.data_kind == "impedance":
        if c.axis != "frequency" or c.axis_unit not in {"Hz", "kHz", "MHz", "GHz"} or imag_column is None:
            raise ValueError("Impedance input requires frequency units and real/imaginary columns")
        if not c.positive_trailing:
            raise ValueError("Frequency data require a positive-trailing delay convention; use Fourier exponent for transform sign")
        axis *= {"Hz": 1., "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[c.axis_unit]
        imag = table[:, imag_column]*scale
        if term.plane == "z" and not c.longitudinal_positive_loss:
            imag = -imag
        transfer = values+1j*imag
        if term.plane != "z":
            transfer /= {"i": 1j, "-i": -1j, "1": 1.}[c.transverse_impedance_factor]
        if c.fourier_exponent == 1:
            transfer = transfer.conj()
        z = transfer if term.plane == "z" else 1j*transfer
        model = SpectrumWakeModel(ImpedanceSpectrum(axis, z.real, z.imag, term.plane == "z"), reconstruction)
    else:
        if imag_column is not None:
            raise ValueError("A real wake function does not take an imaginary column")
        scales = {"time": {"s": 1., "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12},
                  "distance": {"m": 1., "cm": .01, "mm": .001}}
        if c.axis not in scales or c.axis_unit not in scales[c.axis]:
            raise ValueError("Wake function axis must specify time or distance units")
        axis *= scales[c.axis][c.axis_unit]
        if c.axis == "distance":
            axis /= c.reference_beta*const.c
        if not c.positive_trailing:
            axis = -axis
        # Accept either file direction, but reject unordered/duplicate samples.
        if np.all(np.diff(axis) < 0):
            axis, values = axis[::-1], values[::-1]
        model = TabulatedWakeModel(axis, values, causal=causal)
    model.input_metadata = {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                            "format": format, "convention": asdict(c), "columns": columns}
    return model


def impedance_to_wake(spectrum, *, reconstruction="two_sided"):
    from .wake_spectrum import ImpedanceSpectrum, SpectrumWakeModel
    if not isinstance(spectrum, ImpedanceSpectrum):
        raise TypeError("Supply an ImpedanceSpectrum with canonical SI units and Fourier convention")
    return SpectrumWakeModel(spectrum, reconstruction)
