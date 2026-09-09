"""Common result protocol for transverse field solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldResult:
    """Potential and longitudinally integrated transverse fields.

    ``potential`` has units V m and ``integrated_ex``/``integrated_ey`` have
    units V when the supplied source density is in C/m^2.
    ``potential`` is None only when an optional field-only solve omits it.
    """

    potential: np.ndarray | None
    integrated_ex: np.ndarray
    integrated_ey: np.ndarray

    @property
    def ex(self) -> np.ndarray:
        return self.integrated_ex

    @property
    def ey(self) -> np.ndarray:
        return self.integrated_ey
