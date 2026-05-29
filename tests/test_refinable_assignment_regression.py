"""Regression-against-ty: confirm ty still emits the false positive we're
suppressing for iommi refinable declarations like ``attr: str = Refinable()``.

The test fails if ty stops producing ``invalid-assignment`` for such
declarations — the signal that the refinable-assignment suppression in
:mod:`iommi_lsp.analyzers.iommi.analyzer` can be removed.

Skipped if ``ty`` isn't on PATH or ``iommi`` isn't importable for ty.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest


TY_BIN = shutil.which("ty")
pytestmark = pytest.mark.skipif(
    TY_BIN is None,
    reason="real ty binary not on PATH; skipping regression check",
)


REFINABLE_FIXTURE = """\
from iommi.refinable import Refinable, EvaluatedRefinable, SpecialEvaluatedRefinable


class Thing:
    attr: str = Refinable()
    header: dict = EvaluatedRefinable()
    special: int = SpecialEvaluatedRefinable()
"""


def test_ty_still_flags_refinable_assignment(tmp_path):
    """If this stops failing on annotated refinable fields, delete the hijack."""
    assert TY_BIN is not None
    f = tmp_path / "refinable_sample.py"
    f.write_text(REFINABLE_FIXTURE)

    result = subprocess.run(
        [TY_BIN, "check", str(f)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (result.stdout or "") + (result.stderr or "")

    if "Cannot resolve" in combined or "unresolved-import" in combined:
        pytest.skip("iommi not importable for ty; skipping regression check")

    assert "invalid-assignment" in combined, (
        "ty no longer emits invalid-assignment for annotated refinable "
        "declarations — the refinable-assignment suppression in "
        f"IommiAnalyzer can be removed.\noutput:\n{combined}"
    )
    assert "Refinable" in combined, (
        "ty's invalid-assignment text for refinable declarations no longer "
        "mentions Refinable — update or remove "
        f"_is_refinable_assignment_message.\noutput:\n{combined}"
    )
