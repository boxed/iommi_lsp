"""Tests for TemplateAnalyzer — template-name completion in string literals."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iommi_lsp.analyzers.templates import TemplateAnalyzer, discover_templates


def _write_templates(root: Path, *names: str) -> None:
    for rel in names:
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("")


def _write_with_cursor(
    tmp_path: Path, src_before: str, src_after: str = ""
) -> tuple[str, dict]:
    f = tmp_path / "u.py"
    f.write_text(src_before + src_after)
    line = src_before.count("\n")
    last_nl = src_before.rfind("\n")
    character = len(src_before) - (last_nl + 1)
    return f.as_uri(), {"line": line, "character": character}


def _labels(result) -> list[str]:
    return [it["label"] for it in result.items]


@pytest.fixture
def analyzer(tmp_path: Path) -> TemplateAnalyzer:
    _write_templates(
        tmp_path,
        "myapp/templates/myapp/index.html",
        "myapp/templates/myapp/detail.html",
        "blog/templates/blog/post.html",
        "templates/base.html",
    )
    a = TemplateAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    return a


# ---------------------------------------------------------------------------
# discover_templates
# ---------------------------------------------------------------------------


def test_discover_collects_app_templates(tmp_path: Path) -> None:
    _write_templates(
        tmp_path,
        "myapp/templates/myapp/index.html",
        "myapp/templates/myapp/detail.html",
    )
    found = discover_templates(tmp_path)
    assert found == {"myapp/index.html", "myapp/detail.html"}


def test_discover_collects_top_level_templates_dir(tmp_path: Path) -> None:
    _write_templates(tmp_path, "templates/base.html", "templates/foo/bar.html")
    assert discover_templates(tmp_path) == {"base.html", "foo/bar.html"}


def test_discover_skips_venv_and_node_modules(tmp_path: Path) -> None:
    _write_templates(
        tmp_path,
        "app/templates/app/keep.html",
        ".venv/lib/templates/skip.html",
        "node_modules/pkg/templates/skip.html",
    )
    assert discover_templates(tmp_path) == {"app/keep.html"}


def test_discover_skips_hidden_files(tmp_path: Path) -> None:
    _write_templates(tmp_path, "app/templates/app/.hidden", "app/templates/app/v.html")
    assert discover_templates(tmp_path) == {"app/v.html"}


# ---------------------------------------------------------------------------
# completions
# ---------------------------------------------------------------------------


def test_no_completions_without_slash_in_partial(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "x = 'myapp")
    result = analyzer.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


def test_completions_with_slash_partial(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "x = 'myapp/")
    result = analyzer.completions(uri, pos)
    # Exclusive once we have matches — keeps the editor's own filesystem
    # path completion from backfilling unrelated workspace files.
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "myapp/index.html" in labels
    assert "myapp/detail.html" in labels
    # Other apps' templates are filtered out by prefix.
    assert "blog/post.html" not in labels


def test_completions_filtered_by_full_prefix(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "x = 'myapp/in")
    result = analyzer.completions(uri, pos)
    assert _labels(result) == ["myapp/index.html"]
    assert result.items[0]["insertText"] == "myapp/index.html"


def test_completion_textedit_replaces_full_partial(analyzer, tmp_path: Path) -> None:
    # Without an explicit textEdit range, editors that treat `/` as a
    # word boundary (Helix, Neovim built-in client) only replace the
    # trailing word, producing ``myapp/myapp/index.html``. The range
    # must cover from immediately after the opening quote to the cursor.
    src = "x = 'myapp/in"
    uri, pos = _write_with_cursor(tmp_path, src)
    result = analyzer.completions(uri, pos)
    item = result.items[0]
    edit = item["textEdit"]
    assert edit["newText"] == "myapp/index.html"
    # Opening quote sits at column len("x = '") - 1; partial starts one
    # past it. Cursor is at end of src on line 0.
    quote_col = src.index("'")
    assert edit["range"] == {
        "start": {"line": 0, "character": quote_col + 1},
        "end": {"line": 0, "character": len(src)},
    }


def test_completion_textedit_range_inside_call(analyzer, tmp_path: Path) -> None:
    src = "render(request, 'myapp/in"
    uri, pos = _write_with_cursor(tmp_path, src)
    result = analyzer.completions(uri, pos)
    edit = result.items[0]["textEdit"]
    quote_col = src.index("'")
    assert edit["range"] == {
        "start": {"line": 0, "character": quote_col + 1},
        "end": {"line": 0, "character": len(src)},
    }


def test_completion_textedit_range_below_docstring(analyzer, tmp_path: Path) -> None:
    # Cursor on line 2; range must reference line 2, not the absolute
    # file offset.
    pre = '"""Module docstring."""\n\n'
    line_text = "render(request, 'myapp/in"
    uri, pos = _write_with_cursor(tmp_path, pre + line_text)
    result = analyzer.completions(uri, pos)
    edit = result.items[0]["textEdit"]
    quote_col = line_text.index("'")
    assert edit["range"] == {
        "start": {"line": 2, "character": quote_col + 1},
        "end": {"line": 2, "character": len(line_text)},
    }


def test_no_match_for_slash_partial_is_non_exclusive(analyzer, tmp_path: Path) -> None:
    # `/`-containing strings aren't always template references (URLs,
    # filesystem paths, regex). When we have nothing to offer, stay out
    # of the way so ty's items aren't suppressed.
    uri, pos = _write_with_cursor(tmp_path, "x = 'no/such/prefix/")
    result = analyzer.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


def test_completions_double_quote_string(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, 'x = "myapp/')
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "myapp/index.html" in labels


def test_completions_inside_call(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "render(request, 'myapp/")
    labels = set(_labels(analyzer.completions(uri, pos)))
    assert "myapp/index.html" in labels


def test_no_completions_outside_string(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "x = myapp/")
    result = analyzer.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


def test_no_completions_in_triple_quoted_string(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, 'x = """myapp/')
    result = analyzer.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


def test_completions_fire_below_a_docstring(analyzer, tmp_path: Path) -> None:
    # A module-level docstring is the common case — the original scanner
    # bailed the moment it saw `\"\"\"` anywhere before the cursor, which
    # killed completion across the whole file. Make sure the scanner
    # picks up an unrelated single-line string later on.
    uri, pos = _write_with_cursor(
        tmp_path,
        '"""Module docstring."""\n\nrender(request, \'myapp/',
    )
    result = analyzer.completions(uri, pos)
    assert result.exclusive is True
    labels = set(_labels(result))
    assert "myapp/index.html" in labels


def test_no_completions_after_comment_hash(analyzer, tmp_path: Path) -> None:
    uri, pos = _write_with_cursor(tmp_path, "x = 1  # myapp/")
    result = analyzer.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


def test_no_completions_in_multiline_string(analyzer, tmp_path: Path) -> None:
    # The slash sits after a newline inside an open single-quoted string —
    # not legal Python and not a real "single-line" literal. Bail.
    uri, pos = _write_with_cursor(tmp_path, "x = 'foo\nmyapp/")
    result = analyzer.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


def test_empty_index_returns_empty(tmp_path: Path) -> None:
    a = TemplateAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    uri, pos = _write_with_cursor(tmp_path, "x = 'myapp/")
    result = a.completions(uri, pos)
    assert result.items == []
    assert result.exclusive is False


def test_uses_text_provider_over_disk(tmp_path: Path) -> None:
    _write_templates(tmp_path, "app/templates/app/page.html")
    docs: dict[str, str] = {}
    a = TemplateAnalyzer(
        workspace_root=tmp_path,
        text_provider=lambda uri: docs.get(uri),
    )
    asyncio.run(a.index(tmp_path))
    f = tmp_path / "u.py"
    f.write_text("# nothing here\n")
    uri = f.as_uri()
    # The on-disk file lacks the in-progress edit; the editor's buffer
    # has it. Without the text_provider lookup we wouldn't see the partial.
    docs[uri] = "x = 'app/"
    pos = {"line": 0, "character": len("x = 'app/")}
    labels = set(_labels(a.completions(uri, pos)))
    assert "app/page.html" in labels
