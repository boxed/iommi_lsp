"""Tests for the ``f_operator`` suppression in :class:`DjangoAnalyzer`.

ty doesn't see Django's ``Combinable`` overloads for arithmetic / unary
operators, so it emits ``unsupported-operator`` whenever the user does
``datetime - F('x')``, ``F('a') + timedelta(...)``, ``-F('x')``, etc.
Every example here is valid Django (see ``docs/django.com/ref/models/
expressions.html``); the filter is supposed to drop ty's complaint
while still letting genuine "you can't subtract two unrelated values"
bugs through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iommi_lsp.analyzers.django import DjangoAnalyzer, build_index


CORPUS = Path(__file__).parent / "corpus"


def _op_diag(line: int, col_start: int, col_end: int, message: str = "Operator `-` is not supported between objects of type `datetime` and `F`"):
    """Build an ``unsupported-operator`` diagnostic at the given range."""
    return {
        "code": "unsupported-operator",
        "message": message,
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 1,
        "source": "ty",
    }


@pytest.fixture
def analyzer(tmp_path: Path) -> DjangoAnalyzer:
    # The f_operator filter is purely syntactic — it walks the buffer
    # AST and doesn't consult the model index. A tmp-path workspace with
    # no models is fine.
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)
    return a


def _range_of(src: str, expr_substring: str, line_idx: int) -> tuple[int, int, int]:
    line = src.splitlines()[line_idx]
    start = line.index(expr_substring)
    return line_idx, start, start + len(expr_substring)


# ---------------------------------------------------------------------------
# Arithmetic between F() and concrete Python values
# ---------------------------------------------------------------------------


def test_datetime_minus_f_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``datetime.now() - F('pub_date')`` — ty checks ``datetime.__sub__``
    and fails. Python falls back to ``F.__rsub__`` at runtime."""
    src = (
        "from datetime import datetime\n"
        "from django.db.models import F\n"
        "\n"
        "x = datetime.now() - F('pub_date')\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "datetime.now() - F('pub_date')", 3)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_f_plus_timedelta_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``F('pub_date') + timedelta(days=3)`` — the standard Django idiom
    for filtering on a date offset."""
    src = (
        "from datetime import timedelta\n"
        "from django.db.models import F\n"
        "\n"
        "x = F('pub_date') + timedelta(days=3)\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('pub_date') + timedelta(days=3)", 3)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_f_minus_int_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import F\n"
        "x = F('stories_filed') - 1\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('stories_filed') - 1", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_int_minus_f_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """Reversed: ``1 - F('x')`` — Python tries ``int.__sub__`` first,
    falls back to ``F.__rsub__``. ty doesn't model the fallback."""
    src = (
        "from django.db.models import F\n"
        "x = 1 - F('stories_filed')\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "1 - F('stories_filed')", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_f_times_two_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import F\n"
        "x = F('num_chairs') * 2\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('num_chairs') * 2", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_f_div_decimal_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``F('price') / Decimal('1.21')`` — Decimal/F mixing, also fine."""
    src = (
        "from decimal import Decimal\n"
        "from django.db.models import F\n"
        "x = F('price') / Decimal('1.21')\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('price') / Decimal('1.21')", 2)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_f_modulo_int_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import F\n"
        "x = F('id') % 10\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('id') % 10", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_f_pow_int_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import F\n"
        "x = F('side') ** 2\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('side') ** 2", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


# ---------------------------------------------------------------------------
# F-only and chained Combinable arithmetic
# ---------------------------------------------------------------------------


def test_f_plus_f_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import F\n"
        "x = F('a') + F('b')\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('a') + F('b')", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_chained_combined_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``F('a') + F('b') - 1`` — left-associative; ty might flag the
    outer ``- 1`` even though the inner BinOp is already a Combinable."""
    src = (
        "from django.db.models import F\n"
        "x = F('a') + F('b') - 1\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('a') + F('b') - 1", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------


def test_neg_f_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``-F('x')`` calls ``F.__neg__`` which returns ``F * -1`` server-side."""
    src = (
        "from django.db.models import F\n"
        "x = -F('balance')\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "-F('balance')", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_invert_f_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``~F('is_active')`` calls ``F.__invert__`` -> NegatedExpression."""
    src = (
        "from django.db.models import F\n"
        "x = ~F('is_active')\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "~F('is_active')", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


# ---------------------------------------------------------------------------
# Other Combinable factories
# ---------------------------------------------------------------------------


def test_value_plus_f_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``Value(1) + F('x')`` — explicit Value-wrapping is also valid."""
    src = (
        "from django.db.models import F, Value\n"
        "x = Value(1) + F('counter')\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "Value(1) + F('counter')", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_count_minus_f_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """Aggregates are Combinables too."""
    src = (
        "from django.db.models import F, Count\n"
        "x = Count('comments') - F('threshold')\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "Count('comments') - F('threshold')", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_dotted_models_f_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``models.F('x') + timedelta(days=3)`` — accessed via the module."""
    src = (
        "from datetime import timedelta\n"
        "from django.db import models\n"
        "\n"
        "x = models.F('pub_date') + timedelta(days=3)\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "models.F('pub_date') + timedelta(days=3)", 3)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_outerref_plus_f_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    src = (
        "from django.db.models import F, OuterRef\n"
        "x = OuterRef('parent') + F('child')\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "OuterRef('parent') + F('child')", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


# ---------------------------------------------------------------------------
# Combinable method calls that preserve Combinable-ness
# ---------------------------------------------------------------------------


def test_bitand_chained_with_arithmetic_is_dropped(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``F('x').bitand(7) + 1`` — ``bitand`` returns CombinedExpression."""
    src = (
        "from django.db.models import F\n"
        "x = F('flags').bitand(7) + 1\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('flags').bitand(7) + 1", 1)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


# ---------------------------------------------------------------------------
# Negative cases — diagnostics that should *not* be suppressed
# ---------------------------------------------------------------------------


def test_plain_datetime_minus_int_is_kept(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``datetime.now() - 1`` is genuinely broken — no F() anywhere."""
    src = (
        "from datetime import datetime\n"
        "x = datetime.now() - 1\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "datetime.now() - 1", 1)
    diag = _op_diag(line, s, e, message="Operator `-` is not supported between objects of type `datetime` and `int`")
    assert analyzer.is_false_positive(f.as_uri(), diag) is False


def test_string_plus_int_is_kept(analyzer: DjangoAnalyzer, tmp_path: Path):
    """``'hi' + 1`` — no Combinable in sight."""
    src = "x = 'hi' + 1\n"
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "'hi' + 1", 0)
    diag = _op_diag(line, s, e, message="Operator `+` is not supported between objects of type `str` and `int`")
    assert analyzer.is_false_positive(f.as_uri(), diag) is False


def test_local_variable_f_alias_is_kept(analyzer: DjangoAnalyzer, tmp_path: Path):
    """We don't follow data flow — ``e = F('x'); datetime.now() - e`` is
    a known false negative. If we ever start following the binding, flip
    this to ``True``."""
    src = (
        "from datetime import datetime\n"
        "from django.db.models import F\n"
        "\n"
        "e = F('pub_date')\n"
        "x = datetime.now() - e\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "datetime.now() - e", 4)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is False


def test_unsupported_operator_without_f_is_kept(analyzer: DjangoAnalyzer, tmp_path: Path):
    """A real bug in an expression that *also* uses F elsewhere: only the
    BinOp containing F should be suppressed, not unrelated lines."""
    src = (
        "from datetime import datetime\n"
        "from django.db.models import F\n"
        "\n"
        "good = F('a') + F('b')\n"
        "bad = datetime.now() - 'oops'\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "datetime.now() - 'oops'", 4)
    diag = _op_diag(line, s, e)
    assert analyzer.is_false_positive(f.as_uri(), diag) is False


# ---------------------------------------------------------------------------
# Diagnostic-shape / config plumbing
# ---------------------------------------------------------------------------


def test_non_operator_code_is_not_handled_here(analyzer: DjangoAnalyzer, tmp_path: Path):
    """A different code (e.g. ``invalid-argument-type``) shouldn't fall
    through into the f_operator filter even if F() is on the line."""
    src = (
        "from django.db.models import F\n"
        "x = F('a') + 1\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('a') + 1", 1)
    diag = {
        "code": "invalid-argument-type",
        "message": "something else",
        "range": {
            "start": {"line": line, "character": s},
            "end": {"line": line, "character": e},
        },
        "severity": 1,
        "source": "ty",
    }
    assert analyzer.is_false_positive(f.as_uri(), diag) is False


def test_code_as_dict_value_is_handled(analyzer: DjangoAnalyzer, tmp_path: Path):
    """LSP allows ``code`` to be a ``{value, target}`` object — same path."""
    src = (
        "from django.db.models import F\n"
        "x = F('a') + 1\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line, s, e = _range_of(src, "F('a') + 1", 1)
    diag = {
        "code": {"value": "unsupported-operator", "target": "https://example/"},
        "message": "Operator `+` is not supported between objects of type `F` and `int`",
        "range": {
            "start": {"line": line, "character": s},
            "end": {"line": line, "character": e},
        },
        "severity": 1,
        "source": "ty",
    }
    assert analyzer.is_false_positive(f.as_uri(), diag) is True


def test_f_operator_rule_disabled_keeps_diag(tmp_path: Path):
    """When ``f_operator`` is in ``disabled_rules`` we leave the diagnostic
    alone (the user opted into seeing ty's complaints)."""
    from iommi_lsp.config import Config

    src = (
        "from django.db.models import F\n"
        "x = F('a') + 1\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(
        workspace_root=tmp_path,
        config=Config(disabled_rules=frozenset({"f_operator"})),
    )
    a.django_index = build_index(tmp_path)

    line, s, e = _range_of(src, "F('a') + 1", 1)
    diag = _op_diag(line, s, e)
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_range_pointing_only_at_operator_still_matches(analyzer: DjangoAnalyzer, tmp_path: Path):
    """ty may emit a range covering just the operator token rather than
    the whole BinOp — our enclosing-node search has to handle both."""
    src = (
        "from datetime import datetime\n"
        "from django.db.models import F\n"
        "\n"
        "x = datetime.now() - F('pub_date')\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line_text = src.splitlines()[3]
    minus_col = line_text.index(" - ") + 1
    diag = _op_diag(3, minus_col, minus_col + 1)
    assert analyzer.is_false_positive(f.as_uri(), diag) is True
