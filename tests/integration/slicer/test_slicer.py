from __future__ import annotations

import pytest

from .analyse import assert_outputs
from .make_input import CASES
from .run import run_case


@pytest.mark.parametrize("case", sorted(CASES))
def test_slicer_case(case, backend):
    print(f"\n=== Slicer case: {case}, backend={backend} ===")
    output_dir = run_case(case, backend)
    model = CASES[case]["slice_model"]
    assert_outputs(output_dir, model, CASES[case]["num_slices"])
