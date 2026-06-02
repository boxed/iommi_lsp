"""Tests for the ``decorator_typevar`` suppression in :class:`DjangoAnalyzer`.

django-stubs declares ``require_POST`` / ``require_GET`` / ``require_safe`` as
module variables of type ``Callable[[_F], _F]`` — a free ``TypeVar`` at module
scope. ty can't bind that ``_F`` when the variable decorates a view, so it
emits ``invalid-argument-type``: *Argument is incorrect: Expected `TypeVar`,
found `def view(...) -> ...```. There's nothing the user can do (annotating the
return doesn't help), so the filter drops it whenever the flagged line really
is a decorator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iommi_lsp.analyzers.django import DjangoAnalyzer, build_index


# The view-decorator message as ty's LSP publishes it (caret annotation folded
# into the message after the headline).
_MSG = (
    "Argument is incorrect: Expected `TypeVar`, "
    "found `def grab_change_request(request) -> Unknown`"
)


def _arg_diag(line: int, message: str = _MSG, code: str = "invalid-argument-type"):
    """An ``invalid-argument-type`` diagnostic whose range starts on *line*."""
    return {
        "code": code,
        "message": message,
        "range": {
            "start": {"line": line, "character": 0},
            "end": {"line": line, "character": 13},
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


def test_require_post_decorator_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.views.decorators.http import require_POST\n"
        "\n"
        "@require_POST\n"
        "def grab_change_request(request):\n"
        "    return request\n"
    )
    uri = _write(tmp_path, src)
    # Decorator is on source line 3 → 0-indexed range line 2.
    assert analyzer.is_false_positive(uri, _arg_diag(2)) is True


def test_require_get_decorator_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.views.decorators.http import require_GET\n"
        "\n"
        "@require_GET\n"
        "def listing(request):\n"
        "    return request\n"
    )
    uri = _write(tmp_path, src)
    msg = "Argument is incorrect: Expected `TypeVar`, found `def listing(request) -> Unknown`"
    assert analyzer.is_false_positive(uri, _arg_diag(2, message=msg)) is True


def test_non_decorator_line_is_kept(analyzer: DjangoAnalyzer, tmp_path: Path):
    """Same message, but the range points at a plain call line — not a
    decorator. That's a genuine argument-type error; keep it."""
    src = (
        "def takes_typevar(x):\n"
        "    return x\n"
        "\n"
        "result = takes_typevar(takes_typevar)\n"
    )
    uri = _write(tmp_path, src)
    # Line 4 (0-indexed 3) is an ordinary call, no decorator.
    assert analyzer.is_false_positive(uri, _arg_diag(3)) is False


def test_other_argument_error_is_kept(analyzer: DjangoAnalyzer, tmp_path: Path):
    """An ``invalid-argument-type`` that isn't the TypeVar-passthrough shape
    is left alone even on a decorator line."""
    src = (
        "@some_decorator(42)\n"
        "def view(request):\n"
        "    return request\n"
    )
    uri = _write(tmp_path, src)
    diag = _arg_diag(0, message="Argument is incorrect: Expected `str`, found `int`")
    assert analyzer.is_false_positive(uri, diag) is False


def test_wrong_code_is_kept(analyzer: DjangoAnalyzer, tmp_path: Path):
    src = (
        "@require_POST\n"
        "def view(request):\n"
        "    return request\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer.is_false_positive(uri, _arg_diag(0, code="something-else")) is False


def test_disabled_rule_keeps_diagnostic(tmp_path: Path):
    from iommi_lsp.config import Config

    a = DjangoAnalyzer(
        workspace_root=tmp_path,
        config=Config(disabled_rules=frozenset({"decorator_typevar"})),
    )
    a.django_index = build_index(tmp_path)
    src = (
        "@require_POST\n"
        "def view(request):\n"
        "    return request\n"
    )
    f = tmp_path / "views.py"
    f.write_text(src)
    assert a.is_false_positive(f.as_uri(), _arg_diag(0)) is False
