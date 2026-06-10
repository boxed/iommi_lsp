"""Tests for the ``response_header_assignment`` suppression in
:class:`DjangoAnalyzer`.

``HttpResponseBase.__setitem__`` stringifies whatever value it's handed, so
``response[header] = value`` is runtime-valid for any value type. django-stubs
types the setter as ``(str, str | bytes | int)``, so ty flags
``response['Content-Type'] = guess_type(path)[0]`` (a ``str | None``) with
``invalid-assignment``. The filter drops it when the receiver is a Django
response type and the flagged node really is a subscript assignment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iommi_lsp.analyzers.django import DjangoAnalyzer, build_index


# ty's published message for ``response['Content-Type'] = guess_type(path)[0]``.
_MSG = (
    "Invalid subscript assignment with key of type `Literal[\"Content-Type\"]` "
    "and value of type `str | None` on object of type `HttpResponse`"
)


def _diag(line: int, col: int = 4, message: str = _MSG, code: str = "invalid-assignment"):
    return {
        "code": code,
        "message": message,
        "range": {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + 45},
        },
        "severity": 1,
        "source": "ty",
    }


@pytest.fixture
def analyzer(tmp_path: Path) -> DjangoAnalyzer:
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)
    return a


def _write(tmp_path: Path, src: str) -> str:
    f = tmp_path / "views.py"
    f.write_text(src)
    return f.as_uri()


_VIEW_SRC = (
    "from mimetypes import guess_type\n"
    "from django.http import HttpResponse\n"
    "\n"
    "def serve(request, path):\n"
    "    response = HttpResponse()\n"
    "    response['Content-Type'] = guess_type(path)[0]\n"
    "    return response\n"
)


def test_response_subscript_assignment_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    uri = _write(tmp_path, _VIEW_SRC)
    # Subscript assignment is on source line 6 → 0-indexed range line 5.
    assert analyzer.is_false_positive(uri, _diag(5)) is True


def test_redirect_subclass_type_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.http import HttpResponseRedirect\n"
        "\n"
        "def serve(request):\n"
        "    response = HttpResponseRedirect('/')\n"
        "    response['X-Foo'] = bar()\n"
        "    return response\n"
    )
    uri = _write(tmp_path, src)
    msg = (
        "Invalid subscript assignment with key of type `Literal[\"X-Foo\"]` "
        "and value of type `str | None` on object of type `HttpResponseRedirect`"
    )
    assert analyzer.is_false_positive(uri, _diag(4, message=msg)) is True


def test_non_response_receiver_is_kept(analyzer: DjangoAnalyzer, tmp_path: Path):
    """Same diagnostic shape, but the receiver is an ordinary dict-like that
    isn't a Django response — keep it."""
    src = (
        "def f(d, path):\n"
        "    d['Content-Type'] = guess_type(path)[0]\n"
    )
    uri = _write(tmp_path, src)
    msg = (
        "Invalid subscript assignment with key of type `Literal[\"Content-Type\"]` "
        "and value of type `str | None` on object of type `MyMapping`"
    )
    assert analyzer.is_false_positive(uri, _diag(1, message=msg)) is False


def test_non_subscript_target_is_kept(analyzer: DjangoAnalyzer, tmp_path: Path):
    """Message names a response type but the flagged line is not a subscript
    assignment — don't swallow it."""
    src = (
        "from django.http import HttpResponse\n"
        "\n"
        "def serve(request):\n"
        "    response = HttpResponse()\n"
        "    return response\n"
    )
    uri = _write(tmp_path, src)
    # Line 4 (0-indexed 3) is a plain assignment, not a subscript store.
    assert analyzer.is_false_positive(uri, _diag(3, col=4)) is False


def test_wrong_code_is_kept(analyzer: DjangoAnalyzer, tmp_path: Path):
    uri = _write(tmp_path, _VIEW_SRC)
    assert analyzer.is_false_positive(uri, _diag(5, code="something-else")) is False


def test_disabled_rule_keeps_diagnostic(tmp_path: Path):
    from iommi_lsp.config import Config

    a = DjangoAnalyzer(
        workspace_root=tmp_path,
        config=Config(disabled_rules=frozenset({"response_header_assignment"})),
    )
    a.django_index = build_index(tmp_path)
    f = tmp_path / "views.py"
    f.write_text(_VIEW_SRC)
    assert a.is_false_positive(f.as_uri(), _diag(5)) is False
