"""Tests for the unloaded-template-tag diagnostic and its {% load %} fix.

Covers tag-registration discovery in ``templatetags/`` modules, the
``django-unloaded-template-tag`` diagnostic, the quick-fix code actions
on the TemplateAnalyzer, and the CodeActionRouter proxy hook pair.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path

import pytest

from iommi_lsp.analyzers.base import Analyzer
from iommi_lsp.analyzers.templates import (
    TemplateAnalyzer,
    discover_templatetag_registrations,
)
from iommi_lsp.analyzers.templates.analyzer import (
    UNLOADED_FILTER_DIAG_CODE,
    UNLOADED_TAG_DIAG_CODE,
)
from iommi_lsp.interceptor import CodeActionRouter


# ---------------------------------------------------------------------------
# discover_templatetag_registrations — tags
# ---------------------------------------------------------------------------


def _write_lib(tmp_path: Path, name: str, source: str) -> None:
    pkg = tmp_path / "shop" / "templatetags"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / f"{name}.py").write_text(textwrap.dedent(source))


def test_discover_tags_all_registration_forms(tmp_path: Path) -> None:
    _write_lib(tmp_path, "shop_tags", """\
        from django import template

        register = template.Library()

        @register.simple_tag
        def price(): ...

        @register.simple_tag(takes_context=True)
        def cart_total(context): ...

        @register.simple_tag(name='vat')
        def _vat_impl(): ...

        @register.tag
        def render_basket(parser, token): ...

        @register.tag('checkout_button')
        def _checkout(parser, token): ...

        @register.inclusion_tag('shop/badge.html')
        def badge(): ...

        @register.inclusion_tag('shop/row.html', name='product_row')
        def _row(): ...

        def _legacy(parser, token): ...
        register.tag('legacy_tag', _legacy)
    """)
    _filters, tags = discover_templatetag_registrations(tmp_path)
    assert tags == {"shop_tags": {
        "price", "cart_total", "vat", "render_basket",
        "checkout_button", "badge", "product_row", "legacy_tag",
    }}


def test_discover_tags_does_not_mix_filters_in(tmp_path: Path) -> None:
    _write_lib(tmp_path, "shop_tags", """\
        from django import template

        register = template.Library()

        @register.filter
        def currency(v): ...

        @register.simple_tag
        def price(): ...
    """)
    filters, tags = discover_templatetag_registrations(tmp_path)
    assert filters == {"shop_tags": {"currency"}}
    assert tags == {"shop_tags": {"price"}}


# ---------------------------------------------------------------------------
# django-unloaded-template-tag diagnostics
# ---------------------------------------------------------------------------


def _analyzer(tmp_path: Path) -> TemplateAnalyzer:
    a = TemplateAnalyzer(workspace_root=tmp_path)
    asyncio.run(a.index(tmp_path))
    return a


def _diags(analyzer: TemplateAnalyzer, tmp_path: Path, source: str) -> list[dict]:
    page = tmp_path / "page.html"
    page.write_text(textwrap.dedent(source))
    out = analyzer.additional_diagnostics(page.as_uri())
    return [d for d in out if d["code"] == UNLOADED_TAG_DIAG_CODE]


def test_unloaded_project_tag_is_flagged(tmp_path: Path) -> None:
    _write_lib(tmp_path, "shop_tags", """\
        from django import template
        register = template.Library()

        @register.simple_tag
        def price(): ...
    """)
    analyzer = _analyzer(tmp_path)
    diags = _diags(analyzer, tmp_path, """\
        <p>{% price %}</p>
    """)
    assert len(diags) == 1
    assert diags[0]["data"] == {"tag": "price", "libraries": ["shop_tags"]}
    assert "{% load shop_tags %}" in diags[0]["message"]
    # Range points at the tag name itself.
    assert diags[0]["range"]["start"] == {"line": 0, "character": 6}
    assert diags[0]["range"]["end"] == {"line": 0, "character": 11}


def test_loaded_project_tag_is_not_flagged(tmp_path: Path) -> None:
    _write_lib(tmp_path, "shop_tags", """\
        from django import template
        register = template.Library()

        @register.simple_tag
        def price(): ...
    """)
    analyzer = _analyzer(tmp_path)
    assert _diags(analyzer, tmp_path, """\
        {% load shop_tags %}
        <p>{% price %}</p>
    """) == []


def test_static_without_load_is_flagged(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    diags = _diags(analyzer, tmp_path, """\
        <img src="{% static 'logo.png' %}">
    """)
    assert len(diags) == 1
    assert diags[0]["data"] == {"tag": "static", "libraries": ["static"]}


def test_builtin_and_closing_tags_are_not_flagged(tmp_path: Path) -> None:
    _write_lib(tmp_path, "shop_tags", """\
        from django import template
        register = template.Library()

        @register.simple_tag
        def price(): ...
    """)
    analyzer = _analyzer(tmp_path)
    assert _diags(analyzer, tmp_path, """\
        {% if x %}
          {% for y in z %}{{ y }}{% empty %}-{% endfor %}
        {% else %}
          {% csrf_token %}
        {% endif %}
    """) == []


def test_unknown_tag_stays_silent(tmp_path: Path) -> None:
    # ``crispy`` comes from a site-packages library we never index —
    # can't attribute it, so no diagnostic.
    analyzer = _analyzer(tmp_path)
    assert _diags(analyzer, tmp_path, """\
        {% load crispy_forms_tags %}
        {% crispy form %}
        {% unknowable_tag %}
    """) == []


def test_verbatim_and_comment_blocks_are_skipped(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    assert _diags(analyzer, tmp_path, """\
        {% verbatim %}{% static 'x.css' %}{% endverbatim %}
        {% comment %}{% static 'y.css' %}{% endcomment %}
    """) == []


def test_tag_in_two_libraries_lists_both(tmp_path: Path) -> None:
    _write_lib(tmp_path, "a_tags", """\
        from django import template
        register = template.Library()

        @register.simple_tag
        def shared(): ...
    """)
    _write_lib(tmp_path, "b_tags", """\
        from django import template
        register = template.Library()

        @register.simple_tag
        def shared(): ...
    """)
    analyzer = _analyzer(tmp_path)
    diags = _diags(analyzer, tmp_path, "{% shared %}\n")
    assert len(diags) == 1
    assert diags[0]["data"]["libraries"] == ["a_tags", "b_tags"]


def test_loading_either_provider_silences(tmp_path: Path) -> None:
    _write_lib(tmp_path, "a_tags", """\
        from django import template
        register = template.Library()

        @register.simple_tag
        def shared(): ...
    """)
    _write_lib(tmp_path, "b_tags", """\
        from django import template
        register = template.Library()

        @register.simple_tag
        def shared(): ...
    """)
    analyzer = _analyzer(tmp_path)
    assert _diags(analyzer, tmp_path, "{% load b_tags %}\n{% shared %}\n") == []


# ---------------------------------------------------------------------------
# django-unloaded-template-filter diagnostics
# ---------------------------------------------------------------------------


def _filter_diags(
    analyzer: TemplateAnalyzer, tmp_path: Path, source: str,
) -> list[dict]:
    page = tmp_path / "page.html"
    page.write_text(textwrap.dedent(source))
    out = analyzer.additional_diagnostics(page.as_uri())
    return [d for d in out if d["code"] == UNLOADED_FILTER_DIAG_CODE]


def test_unloaded_project_filter_is_flagged(tmp_path: Path) -> None:
    _write_lib(tmp_path, "shop_tags", """\
        from django import template
        register = template.Library()

        @register.filter
        def currency(v): ...
    """)
    analyzer = _analyzer(tmp_path)
    diags = _filter_diags(analyzer, tmp_path, """\
        <p>{{ product.price|currency }}</p>
    """)
    assert len(diags) == 1
    assert diags[0]["data"] == {"filter": "currency", "libraries": ["shop_tags"]}
    # Range points at the filter name itself.
    assert diags[0]["range"]["start"] == {"line": 0, "character": 20}
    assert diags[0]["range"]["end"] == {"line": 0, "character": 28}


def test_loaded_project_filter_is_not_flagged(tmp_path: Path) -> None:
    _write_lib(tmp_path, "shop_tags", """\
        from django import template
        register = template.Library()

        @register.filter
        def currency(v): ...
    """)
    analyzer = _analyzer(tmp_path)
    assert _filter_diags(analyzer, tmp_path, """\
        {% load shop_tags %}
        <p>{{ product.price|currency }}</p>
    """) == []


def test_humanize_filter_without_load_is_flagged(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    diags = _filter_diags(analyzer, tmp_path, "{{ count|intcomma }}\n")
    assert len(diags) == 1
    assert diags[0]["data"] == {"filter": "intcomma", "libraries": ["humanize"]}
    assert "{% load humanize %}" in diags[0]["message"]


def test_builtin_filters_are_not_flagged(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    assert _filter_diags(analyzer, tmp_path, """\
        {{ x|upper|truncatewords:3 }}
        {% if y|length > 2 %}{{ y|join:", " }}{% endif %}
    """) == []


def test_pipe_inside_string_literal_is_not_a_filter(tmp_path: Path) -> None:
    _write_lib(tmp_path, "shop_tags", """\
        from django import template
        register = template.Library()

        @register.filter
        def currency(v): ...
    """)
    analyzer = _analyzer(tmp_path)
    # ``a|currency`` inside the string argument is not a filter use.
    assert _filter_diags(analyzer, tmp_path, """\
        {{ x|default:"a|currency" }}
    """) == []


def test_filter_in_tag_body_is_flagged(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    diags = _filter_diags(analyzer, tmp_path, "{% if n|intcomma %}x{% endif %}\n")
    assert len(diags) == 1
    assert diags[0]["data"]["filter"] == "intcomma"


def test_unknown_filter_stays_silent(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    assert _filter_diags(analyzer, tmp_path, "{{ x|some_third_party_filter }}\n") == []


def test_filter_in_verbatim_block_is_skipped(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    assert _filter_diags(analyzer, tmp_path, """\
        {% verbatim %}{{ n|intcomma }}{% endverbatim %}
    """) == []


def test_filter_fix_offers_load_action(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    page = tmp_path / "page.html"
    page.write_text("{{ count|intcomma }}\n")
    uri = page.as_uri()
    actions = analyzer.code_actions(uri, _WHOLE_FILE, {})
    assert len(actions) == 1
    action = actions[0]
    assert action["title"] == "Insert {% load humanize %}"
    assert action["diagnostics"][0]["code"] == UNLOADED_FILTER_DIAG_CODE
    (edit,) = action["edit"]["changes"][uri]
    assert edit["range"]["start"] == {"line": 0, "character": 0}
    assert edit["newText"] == "{% load humanize %}\n"


def test_tag_and_filter_from_same_library_yield_one_action(tmp_path: Path) -> None:
    _write_lib(tmp_path, "shop_tags", """\
        from django import template
        register = template.Library()

        @register.simple_tag
        def price(): ...

        @register.filter
        def currency(v): ...
    """)
    analyzer = _analyzer(tmp_path)
    page = tmp_path / "page.html"
    page.write_text("{% price %} {{ x|currency }}\n")
    uri = page.as_uri()
    actions = analyzer.code_actions(uri, _WHOLE_FILE, {})
    assert len(actions) == 1
    assert actions[0]["title"] == "Insert {% load shop_tags %}"


# ---------------------------------------------------------------------------
# code actions — edit placement
# ---------------------------------------------------------------------------


_WHOLE_FILE = {
    "start": {"line": 0, "character": 0},
    "end": {"line": 9999, "character": 0},
}


def _actions(
    analyzer: TemplateAnalyzer, tmp_path: Path, source: str, rng: dict = _WHOLE_FILE,
) -> tuple[str, list[dict]]:
    page = tmp_path / "page.html"
    page.write_text(textwrap.dedent(source))
    uri = page.as_uri()
    return uri, analyzer.code_actions(uri, rng, {})


def test_fix_inserts_load_at_top_of_plain_file(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    uri, actions = _actions(analyzer, tmp_path, """\
        <img src="{% static 'logo.png' %}">
    """)
    assert len(actions) == 1
    action = actions[0]
    assert action["title"] == "Insert {% load static %}"
    assert action["kind"] == "quickfix"
    assert action["diagnostics"][0]["code"] == UNLOADED_TAG_DIAG_CODE
    (edit,) = action["edit"]["changes"][uri]
    assert edit["range"]["start"] == {"line": 0, "character": 0}
    assert edit["range"]["end"] == {"line": 0, "character": 0}
    assert edit["newText"] == "{% load static %}\n"


def test_fix_inserts_load_after_extends(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    uri, actions = _actions(analyzer, tmp_path, """\
        {% extends "base.html" %}
        {% block content %}{% static 'x.css' %}{% endblock %}
    """)
    assert len(actions) == 1
    (edit,) = actions[0]["edit"]["changes"][uri]
    assert edit["range"]["start"] == {"line": 1, "character": 0}
    assert edit["newText"] == "{% load static %}\n"


def test_fix_appends_to_existing_load_tag(tmp_path: Path) -> None:
    _write_lib(tmp_path, "shop_tags", """\
        from django import template
        register = template.Library()

        @register.simple_tag
        def price(): ...
    """)
    analyzer = _analyzer(tmp_path)
    uri, actions = _actions(analyzer, tmp_path, """\
        {% load static %}
        <p>{% price %}</p>
    """)
    assert len(actions) == 1
    action = actions[0]
    assert action["title"] == "Add 'shop_tags' to {% load %}"
    (edit,) = action["edit"]["changes"][uri]
    # Appended right after ``static`` inside the existing load tag.
    assert edit["range"]["start"] == {"line": 0, "character": 14}
    assert edit["range"]["end"] == {"line": 0, "character": 14}
    assert edit["newText"] == " shop_tags"


def test_fix_skips_from_form_load_tag(tmp_path: Path) -> None:
    # ``{% load price from shop_tags %}`` must not be appended to — a new
    # line is inserted instead.
    _write_lib(tmp_path, "shop_tags", """\
        from django import template
        register = template.Library()

        @register.simple_tag
        def price(): ...
    """)
    analyzer = _analyzer(tmp_path)
    uri, actions = _actions(analyzer, tmp_path, """\
        {% load other from somewhere_else %}
        <img src="{% static 'x.png' %}">
    """)
    assert len(actions) == 1
    (edit,) = actions[0]["edit"]["changes"][uri]
    assert edit["newText"] == "{% load static %}\n"
    assert edit["range"]["start"] == {"line": 0, "character": 0}


def test_no_actions_outside_diagnostic_range(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    rng = {"start": {"line": 5, "character": 0}, "end": {"line": 5, "character": 0}}
    _uri, actions = _actions(analyzer, tmp_path, """\
        <img src="{% static 'logo.png' %}">
    """, rng)
    assert actions == []


def test_cursor_on_diagnostic_yields_action(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    # Zero-width cursor range sitting on the tag name.
    rng = {"start": {"line": 0, "character": 14}, "end": {"line": 0, "character": 14}}
    _uri, actions = _actions(analyzer, tmp_path, """\
        <img src="{% static 'logo.png' %}">
    """, rng)
    assert len(actions) == 1


def test_repeated_uses_of_same_library_yield_one_action(tmp_path: Path) -> None:
    analyzer = _analyzer(tmp_path)
    _uri, actions = _actions(analyzer, tmp_path, """\
        <img src="{% static 'a.png' %}">
        <img src="{% static 'b.png' %}">
    """)
    assert len(actions) == 1


# ---------------------------------------------------------------------------
# CodeActionRouter
# ---------------------------------------------------------------------------


class _CaptureWriter:
    """StreamWriter stub: stores framed writes for later inspection."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, data: bytes) -> None:
        self.buf.extend(data)

    async def drain(self) -> None:
        return None

    def messages(self) -> list[dict]:
        out: list[dict] = []
        view = bytes(self.buf)
        while view:
            header_end = view.find(b"\r\n\r\n")
            if header_end < 0:
                break
            header = view[:header_end].decode("ascii")
            cl = 0
            for line in header.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    cl = int(line.split(":", 1)[1].strip())
            body = view[header_end + 4:header_end + 4 + cl]
            out.append(json.loads(body))
            view = view[header_end + 4 + cl:]
        return out


def _frame(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


class _ActionProvider(Analyzer):
    """Test analyzer with pre-baked code_actions return value."""

    name = "provider"

    def __init__(self, actions: list[dict]) -> None:
        self.actions = actions
        self.calls: list[tuple[str, dict, dict]] = []

    async def index(self, workspace_root: Path) -> None: ...
    async def on_file_changed(self, uri: str) -> None: ...
    def is_false_positive(self, uri, diag): return False

    def code_actions(self, uri: str, range_: dict, context: dict) -> list[dict]:
        self.calls.append((uri, range_, context))
        return list(self.actions)


def _code_action_request(msg_id: int = 7) -> bytes:
    return _frame({
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": "textDocument/codeAction",
        "params": {
            "textDocument": {"uri": "file:///page.html"},
            "range": {
                "start": {"line": 0, "character": 3},
                "end": {"line": 0, "character": 3},
            },
            "context": {"diagnostics": []},
        },
    })


@pytest.mark.asyncio
async def test_code_action_request_short_circuits_when_provider_has_actions():
    action = {"title": "Insert {% load static %}", "kind": "quickfix"}
    provider = _ActionProvider([action])
    writer = _CaptureWriter()
    router = CodeActionRouter(analyzers=[provider])
    router.attach_editor_writer(writer)

    out = await router.on_request(_code_action_request(7))
    assert out is None  # dropped — we answered directly
    assert provider.calls[0][0] == "file:///page.html"

    msgs = writer.messages()
    assert len(msgs) == 1
    assert msgs[0]["id"] == 7
    assert msgs[0]["result"] == [action]


@pytest.mark.asyncio
async def test_code_action_request_forwards_when_no_actions():
    provider = _ActionProvider([])
    writer = _CaptureWriter()
    router = CodeActionRouter(analyzers=[provider])
    router.attach_editor_writer(writer)

    req = _code_action_request(8)
    out = await router.on_request(req)
    assert out is req  # forwarded to ty
    assert writer.messages() == []


@pytest.mark.asyncio
async def test_initialize_response_gets_code_action_capability():
    router = CodeActionRouter(analyzers=[])
    init_req = _frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    await router.on_request(init_req)

    init_resp = _frame({
        "jsonrpc": "2.0", "id": 1,
        "result": {"capabilities": {"hoverProvider": True}},
    })
    out = await router.on_response(init_resp)
    assert out is not None
    decoded = json.loads(out)
    assert decoded["result"]["capabilities"]["codeActionProvider"] == {
        "codeActionKinds": ["quickfix"],
    }
    assert decoded["result"]["capabilities"]["hoverProvider"] is True


@pytest.mark.asyncio
async def test_initialize_response_left_alone_when_ty_already_offers_code_actions():
    router = CodeActionRouter(analyzers=[])
    init_req = _frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    await router.on_request(init_req)

    init_resp = _frame({
        "jsonrpc": "2.0", "id": 1,
        "result": {"capabilities": {"codeActionProvider": True}},
    })
    out = await router.on_response(init_resp)
    assert out == init_resp
