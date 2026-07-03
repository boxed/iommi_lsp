"""Regression-against-ty: confirm ty still emits the false positive we're
suppressing for scalar Django field declarations like
``name: str = CharField()``.

The test fails if ty stops producing ``invalid-assignment`` for such
declarations — the signal that our suppression in
:mod:`iommi_lsp.analyzers.django.analyzer` can be removed.

Skipped if ``ty`` isn't on PATH.
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


SCALAR_FIELD_FIXTURE = """\
from django.db import models


class Thing(models.Model):
    name: str = models.CharField(max_length=100)
    count: int = models.IntegerField()
    ratio: float = models.FloatField()
"""


def test_ty_still_flags_scalar_field_assignment(tmp_path):
    """If this stops failing on annotated scalar fields, delete the hijack."""
    assert TY_BIN is not None
    f = tmp_path / "scalar_sample.py"
    f.write_text(SCALAR_FIELD_FIXTURE)

    result = subprocess.run(
        [TY_BIN, "check", str(f)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (result.stdout or "") + (result.stderr or "")

    assert "invalid-assignment" in combined, (
        "ty no longer emits invalid-assignment for annotated scalar field "
        "declarations — the scalar_field_assignment suppression in "
        f"DjangoAnalyzer can be removed.\noutput:\n{combined}"
    )
    # The message names the field class and the declared scalar target —
    # both anchors _is_scalar_field_assignment_message relies on.
    assert "CharField" in combined and "`str`" in combined, (
        "ty's invalid-assignment text for scalar field declarations no "
        "longer names the field type / target scalar — update or remove "
        f"_is_scalar_field_assignment_message.\noutput:\n{combined}"
    )
