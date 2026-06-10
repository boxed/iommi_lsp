"""Regression-against-ty: confirm ty still emits the false positive we're
suppressing for ``response[header] = value`` assignments on Django responses.

The test fails if ty stops producing ``invalid-assignment`` for such
subscript assignments — the signal that our suppression in
:mod:`iommi_lsp.analyzers.django.analyzer` can be removed.

Skipped if ``ty`` isn't on PATH or django-stubs isn't importable in the
environment ty resolves (without the stub, ``__setitem__`` is untyped and ty
emits nothing).
"""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess

import pytest


TY_BIN = shutil.which("ty")
# The value-type constraint on ``__setitem__`` comes from django-stubs; without
# it ty sees the untyped runtime signature and emits nothing, so there's no
# false positive to regress against.
try:
    importlib.metadata.distribution("django-stubs")
    HAS_DJANGO_STUBS = True
except importlib.metadata.PackageNotFoundError:
    HAS_DJANGO_STUBS = False
pytestmark = [
    pytest.mark.skipif(
        TY_BIN is None,
        reason="real ty binary not on PATH; skipping regression check",
    ),
    pytest.mark.skipif(
        not HAS_DJANGO_STUBS,
        reason="django-stubs not installed; ty can't see the typed __setitem__",
    ),
]


RESPONSE_HEADER_FIXTURE = """\
from mimetypes import guess_type

from django.http import HttpResponse


def serve(request, path):
    response = HttpResponse()
    response['Content-Type'] = guess_type(path)[0]
    return response
"""


def test_ty_still_flags_response_header_assignment(tmp_path):
    """If this stops failing, delete the response_header_assignment hijack."""
    assert TY_BIN is not None
    f = tmp_path / "response_sample.py"
    f.write_text(RESPONSE_HEADER_FIXTURE)

    result = subprocess.run(
        [TY_BIN, "check", str(f)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (result.stdout or "") + (result.stderr or "")

    if "django" in combined.lower() and "unresolved-import" in combined:
        pytest.skip("django/django-stubs not resolvable by ty in this env")

    assert "invalid-assignment" in combined, (
        "ty no longer emits invalid-assignment for `response[header] = value` "
        "— the response_header_assignment suppression in DjangoAnalyzer can "
        f"be removed.\noutput:\n{combined}"
    )
    assert "subscript assignment" in combined, (
        "ty's invalid-assignment text for response-header assignment no longer "
        "says 'subscript assignment' — update or remove "
        f"_is_response_subscript_assignment_message.\noutput:\n{combined}"
    )
