"""Canonical wake file entry point and explicit external-data conventions."""
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
    distance_convention: Literal["beta_c_tau", "c_tau"] = "beta_c_tau"

    def __post_init__(self):
        import math
        if self.distance_convention not in {"beta_c_tau", "c_tau"}:
            raise ValueError("Distance convention must be beta_c_tau or c_tau")
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


def read_wake_file(path, *, component=None, spatial=None, format="tfs", reconstruction="two_sided"):
    """Read canonical wake TFS; external numeric files need explicit conversion.

    Units, signs, source normalization and temporal causality belong to the
    file header. Spectrum reconstruction remains a tracking-model choice.
    """
    from .wake_tfs import read_wake_tfs
    if format != "tfs":
        raise ValueError("Wake tracking only accepts canonical TFS; convert external files with PASS.tool.wake_conversion")
    return read_wake_tfs(path, component=component, spatial=spatial, reconstruction=reconstruction)


def impedance_to_wake(spectrum, *, reconstruction="two_sided"):
    from .wake_spectrum import ImpedanceSpectrum, SpectrumWakeModel
    if not isinstance(spectrum, ImpedanceSpectrum):
        raise TypeError("Supply an ImpedanceSpectrum with canonical SI units and Fourier convention")
    return SpectrumWakeModel(spectrum, reconstruction)
