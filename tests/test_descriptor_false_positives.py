"""The false positives a real workspace run turned up, one test each.

Everything here was found by pointing the proxy at a large Django +
iommi project and reading what survived the filter: Django's field
descriptors are invisible to ty, so anything reached *through* a field
(an attribute, an assignment, an operator, a narrowing) gets reported,
and a handful of our own resolvers guessed wrong about which class a
name referred to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iommi_lsp.analyzers.django import DjangoAnalyzer, build_index
from iommi_lsp.config import Config


CORPUS = Path(__file__).parent / "corpus"


@pytest.fixture
def analyzer_basic() -> DjangoAnalyzer:
    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")
    return a


@pytest.fixture
def analyzer_blog() -> DjangoAnalyzer:
    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")
    return a


@pytest.fixture
def analyzer_two_apps() -> DjangoAnalyzer:
    a = DjangoAnalyzer(workspace_root=CORPUS / "two_apps")
    a.django_index = build_index(CORPUS / "two_apps")
    return a


def _write(tmp_path: Path, src: str, name: str = "u.py") -> str:
    f = tmp_path / name
    f.write_text(src)
    return f.as_uri()


def _diag(message: str, *, code: str | None, line: int = 0, char: int = 0) -> dict:
    d: dict = {
        "range": {
            "start": {"line": line, "character": char},
            "end": {"line": line, "character": char + 1},
        },
        "message": message,
        "source": "ty",
    }
    if code is not None:
        d["code"] = code
    return d


# -- implicit `id` ---------------------------------------------------------

def test_implicit_id_is_a_valid_lookup(analyzer_basic, tmp_path):
    """``page__id=1`` — Django adds ``id`` when there's no explicit PK."""
    src = (
        "from myapp.models import Profile\n"
        "Profile.objects.filter(user__id=1)\n"
    )
    assert analyzer_basic.additional_diagnostics(_write(tmp_path, src)) == []


def test_id_on_explicit_pk_model_is_still_flagged(analyzer_basic, tmp_path):
    """``WithExplicitPK`` declares ``code`` as the PK, so there's no ``id``."""
    src = (
        "from myapp.models import WithExplicitPK\n"
        "WithExplicitPK.objects.filter(id=1)\n"
    )
    diags = analyzer_basic.additional_diagnostics(_write(tmp_path, src))
    assert len(diags) == 1
    assert "'id'" in diags[0]["message"]


# -- reverse *query* names -------------------------------------------------

def test_reverse_query_name_is_accepted(analyzer_blog, tmp_path):
    """``Comment.article`` has no related_name → filter on ``comment``.

    The ``comment_set`` accessor keeps working too; Django only accepts
    the query name, but flagging the accessor would trade one false
    positive for another.
    """
    src = (
        "from blog.models import Article\n"
        "Article.objects.filter(comment__body='x')\n"
        "Article.objects.filter(comment_set__body='x')\n"
    )
    assert analyzer_blog.additional_diagnostics(_write(tmp_path, src)) == []


def test_reverse_query_name_is_what_completion_offers(analyzer_blog, tmp_path):
    src = "from blog.models import Article\nArticle.objects.filter("
    uri = _write(tmp_path, src)
    pos = {"line": 1, "character": len(src.splitlines()[1])}
    labels = {item["label"] for item in analyzer_blog.completions(uri, pos).items}
    assert "comment" in labels
    assert "comment_set" not in labels


# -- module-qualified model references -------------------------------------

def test_module_alias_picks_the_right_same_named_model(
    analyzer_two_apps, tmp_path
):
    """``other.Service`` is beta's, not the ``Service`` this file imported."""
    src = (
        "import beta.models as other\n"
        "from alpha.models import Service\n"
        "other.Service.objects.filter(label='x')\n"
        "Service.objects.filter(name='x')\n"
    )
    assert analyzer_two_apps.additional_diagnostics(_write(tmp_path, src)) == []


def test_module_alias_still_flags_a_real_typo(analyzer_two_apps, tmp_path):
    src = (
        "import beta.models as other\n"
        "from alpha.models import Service\n"
        "other.Service.objects.filter(name='x')\n"
    )
    diags = analyzer_two_apps.additional_diagnostics(_write(tmp_path, src))
    assert len(diags) == 1
    assert "beta.models.Service" in diags[0]["message"]


def test_class_wording_resolves_the_model(analyzer_basic, tmp_path):
    """ty says ``Class `User``` when the receiver is the class itself.

    The receiver here is an unannotated parameter, so only the type name
    in ty's message says which class it is.
    """
    src = (
        "def field_of(model, attr):\n"
        "    return model._meta.get_field(attr)\n"
    )
    uri = _write(tmp_path, src)
    diag = _diag(
        "Class `User` has no attribute `_meta`",
        code="unresolved-attribute",
        line=1,
        char=len("    return model."),
    )
    assert analyzer_basic.is_false_positive(uri, diag) is True


# -- attribute access through a field --------------------------------------

@pytest.mark.parametrize("message", [
    "Object of type `ForeignKey` has no attribute `project`",
    "Object of type `ForeignKey & ~AlwaysFalsy` has no attribute `template`",
    "Attribute `status` is not defined on `ForeignKey` in union `ForeignKey | Unknown`",
    "Object of type `ManyToManyField` has no attribute `count`",
])
def test_attribute_through_a_relation_field_is_dropped(
    analyzer_basic, tmp_path, message
):
    uri = _write(tmp_path, "x = 1\n")
    assert analyzer_basic.is_false_positive(
        uri, _diag(message, code="unresolved-attribute")
    ) is True


def test_attribute_on_a_plain_class_is_kept(analyzer_basic, tmp_path):
    uri = _write(tmp_path, "x = 1\n")
    assert analyzer_basic.is_false_positive(
        uri,
        _diag(
            "Object of type `Decimal` has no attribute `nope`",
            code="unresolved-attribute",
        ),
    ) is False


# -- assignment onto a field attribute -------------------------------------

@pytest.mark.parametrize("message", [
    "Object of type `None` is not assignable to attribute `rejected_time` of type `DateTimeField`",
    "Object of type `Unknown | None` is not assignable to attribute `depends_on` of type `ForeignKey`",
    "Object of type `str` is not assignable to attribute `price` of type `MoneyField`",
])
def test_assignment_onto_a_field_attribute_is_dropped(
    analyzer_basic, tmp_path, message
):
    uri = _write(tmp_path, "x = 1\n")
    assert analyzer_basic.is_false_positive(
        uri, _diag(message, code="invalid-assignment")
    ) is True


def test_assignment_onto_a_plain_attribute_is_kept(analyzer_basic, tmp_path):
    uri = _write(tmp_path, "x = 1\n")
    assert analyzer_basic.is_false_positive(
        uri,
        _diag(
            "Object of type `None` is not assignable to attribute `total` of type `Decimal`",
            code="invalid-assignment",
        ),
    ) is False


# -- operators on a field-typed operand ------------------------------------

def test_operator_on_a_field_operand_is_dropped(analyzer_basic, tmp_path):
    uri = _write(tmp_path, "x = 1\n")
    message = (
        "Operator `/` is not supported between objects of type "
        "`(Unknown & ~AlwaysFalsy) | Literal[0]` and `FloatField & ~AlwaysFalsy`"
    )
    assert analyzer_basic.is_false_positive(
        uri, _diag(message, code="unsupported-operator")
    ) is True


def test_operator_without_a_field_operand_is_kept(analyzer_basic, tmp_path):
    uri = _write(tmp_path, "x = 1\n")
    message = "Operator `/` is not supported between objects of type `str` and `int`"
    assert analyzer_basic.is_false_positive(
        uri, _diag(message, code="unsupported-operator")
    ) is False


# -- unreachable behind a field-typed narrowing ----------------------------

UNREACHABLE = "Code is unreachable\nThis may depend on your current environment and settings"


def test_unreachable_behind_a_field_guard_is_dropped(analyzer_basic, tmp_path):
    src = (
        "from myapp.models import Profile\n"
        "def f(p: Profile):\n"
        "    if p.bio is None:\n"
        "        raise ValueError('no bio')\n"
        "    return p.bio\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.is_false_positive(
        uri, _diag(UNREACHABLE, code=None, line=4, char=4)
    ) is True


def test_unreachable_behind_a_field_assert_is_dropped(analyzer_basic, tmp_path):
    src = (
        "from myapp.models import Profile\n"
        "def f(p: Profile):\n"
        "    assert p.bio is None\n"
        "    return 1\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.is_false_positive(
        uri, _diag(UNREACHABLE, code=None, line=3, char=4)
    ) is True


def test_unreachable_after_a_plain_return_is_kept(analyzer_basic, tmp_path):
    src = (
        "def f(x):\n"
        "    return 1\n"
        "    return 2\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.is_false_positive(
        uri, _diag(UNREACHABLE, code=None, line=2, char=4)
    ) is False


def test_names_to_types_resolves_an_unannotated_name(tmp_path):
    """The fixture parameter says nothing; ty's own table says it's a Profile."""
    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")
    a.config = Config(names_to_types={"profile": ("myapp.models.Profile",)})
    src = (
        "def test_x(profile):\n"
        "    assert profile.bio is None\n"
        "    assert True\n"
    )
    uri = _write(tmp_path, src, name="t_names__tests.py")
    assert a.is_false_positive(
        uri, _diag(UNREACHABLE, code=None, line=2, char=4)
    ) is True


# -- unused parameters -----------------------------------------------------

def test_unused_fixture_param_is_dropped(analyzer_basic, tmp_path):
    src = "def test_thing(project, action):\n    assert 1\n"
    uri = _write(tmp_path, src, name="a__tests.py")
    assert analyzer_basic.is_false_positive(
        uri, _diag("`action` is unused", code=None, line=0, char=len("def test_thing(project, "))
    ) is True


def test_unused_param_in_a_stub_is_dropped(analyzer_basic, tmp_path):
    src = (
        "class P:\n"
        "    def stream(self, messages):\n"
        "        raise NotImplementedError('later')\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.is_false_positive(
        uri, _diag("`messages` is unused", code=None, line=1, char=len("    def stream(self, "))
    ) is True


def test_unused_param_in_a_real_function_is_kept(analyzer_basic, tmp_path):
    src = (
        "def compute(value, unused):\n"
        "    return value * 2\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.is_false_positive(
        uri, _diag("`unused` is unused", code=None, line=0, char=len("def compute(value, "))
    ) is False


def test_unused_local_in_a_test_is_kept(analyzer_basic, tmp_path):
    """Only parameters are the fixture idiom; a dead local is still dead."""
    src = (
        "def test_thing():\n"
        "    total = 1\n"
        "    assert True\n"
    )
    uri = _write(tmp_path, src, name="b__tests.py")
    assert analyzer_basic.is_false_positive(
        uri, _diag("`total` is unused", code=None, line=1, char=4)
    ) is False


# -- union of model class objects ------------------------------------------

def test_objects_on_a_union_of_model_classes_is_dropped(
    analyzer_basic, tmp_path
):
    src = (
        "from myapp.models import User, Profile\n"
        "for model in (User, Profile):\n"
        "    model.objects.all()\n"
    )
    uri = _write(tmp_path, src)
    message = (
        "Object of type `<class 'User'> | <class 'Profile'>` "
        "has no attribute `objects`"
    )
    assert analyzer_basic.is_false_positive(
        uri, _diag(message, code="unresolved-attribute", line=2, char=len("    model."))
    ) is True


def test_unknown_attr_on_a_union_of_model_classes_is_kept(
    analyzer_basic, tmp_path
):
    src = (
        "from myapp.models import User, Profile\n"
        "for model in (User, Profile):\n"
        "    model.bogus\n"
    )
    uri = _write(tmp_path, src)
    message = (
        "Object of type `<class 'User'> | <class 'Profile'>` "
        "has no attribute `bogus`"
    )
    assert analyzer_basic.is_false_positive(
        uri, _diag(message, code="unresolved-attribute", line=2, char=len("    model."))
    ) is False


def test_operator_between_two_field_operands_is_dropped(analyzer_basic, tmp_path):
    uri = _write(tmp_path, "x = 1\n")
    message = (
        "Operator `+` is not supported between two objects of type `FloatField`"
    )
    assert analyzer_basic.is_false_positive(
        uri, _diag(message, code="unsupported-operator")
    ) is True


# -- a field passed as an argument -----------------------------------------

def test_field_valued_argument_is_dropped(analyzer_basic, tmp_path):
    uri = _write(tmp_path, "x = 1\n")
    message = (
        "Argument to function `_check_limit` is incorrect: "
        "Expected `int`, found `IntegerField`"
    )
    assert analyzer_basic.is_false_positive(
        uri, _diag(message, code="invalid-argument-type")
    ) is True


def test_plain_argument_mismatch_is_kept(analyzer_basic, tmp_path):
    uri = _write(tmp_path, "x = 1\n")
    message = (
        "Argument to function `_check_limit` is incorrect: "
        "Expected `int`, found `str`"
    )
    assert analyzer_basic.is_false_positive(
        uri, _diag(message, code="invalid-argument-type")
    ) is False


# -- function-local imports ------------------------------------------------

def test_function_local_import_shadows_the_module_one(
    analyzer_two_apps, tmp_path
):
    """A deferred import inside a fixture names a different Service."""
    src = (
        "from alpha.models import Service\n"
        "\n"
        "def fixture():\n"
        "    from beta.models import Service\n"
        "    return Service.objects.filter(label='x')\n"
        "\n"
        "def other():\n"
        "    return Service.objects.filter(name='x')\n"
    )
    assert analyzer_two_apps.additional_diagnostics(_write(tmp_path, src)) == []


def test_function_local_import_still_flags_a_typo(analyzer_two_apps, tmp_path):
    src = (
        "from alpha.models import Service\n"
        "\n"
        "def fixture():\n"
        "    from beta.models import Service\n"
        "    return Service.objects.filter(name='x')\n"
    )
    diags = analyzer_two_apps.additional_diagnostics(_write(tmp_path, src))
    assert len(diags) == 1
    assert "beta.models.Service" in diags[0]["message"]


def test_a_nested_functions_import_does_not_leak_outward(
    analyzer_two_apps, tmp_path
):
    """The inner ``def`` binds ``Service`` only for its own body."""
    src = (
        "from alpha.models import Service\n"
        "\n"
        "def outer():\n"
        "    def inner():\n"
        "        from beta.models import Service\n"
        "        return Service.objects.filter(label='x')\n"
        "    return Service.objects.filter(name='x'), inner\n"
    )
    assert analyzer_two_apps.additional_diagnostics(_write(tmp_path, src)) == []


# -- `**dict` splat calls --------------------------------------------------

def test_splat_call_argument_diagnostics_are_dropped(analyzer_basic, tmp_path):
    """One `**conf` call, one diagnostic per parameter of the callee."""
    src = (
        "import sentry_sdk\n"
        "conf = dict(dsn='x', debug=True)\n"
        "sentry_sdk.init(**conf)\n"
    )
    uri = _write(tmp_path, src)
    char = src.splitlines()[2].index("conf", len("sentry_sdk.init(*"))
    for code in ("invalid-argument-type", "no-matching-overload"):
        assert analyzer_basic.is_false_positive(
            uri,
            _diag(
                "Argument to `Client.__init__` is incorrect: "
                "Expected `bool`, found `str | bool`",
                code=code, line=2, char=char,
            ),
        ) is True


def test_spelled_out_call_argument_diagnostics_are_kept(
    analyzer_basic, tmp_path
):
    src = (
        "import sentry_sdk\n"
        "sentry_sdk.init(debug='yes')\n"
    )
    uri = _write(tmp_path, src)
    char = src.splitlines()[1].index("'yes'")
    assert analyzer_basic.is_false_positive(
        uri,
        _diag(
            "Argument to `Client.__init__` is incorrect: "
            "Expected `bool`, found `str`",
            code="invalid-argument-type", line=1, char=char,
        ),
    ) is False


def test_older_ty_wording_resolves_the_model(analyzer_basic, tmp_path):
    """``Unresolved attribute `x` on type `Y``` — another ty spelling."""
    src = (
        "def field_of(model, attr):\n"
        "    return model._meta.get_field(attr)\n"
    )
    uri = _write(tmp_path, src)
    diag = _diag(
        "Unresolved attribute `_meta` on type `User`",
        code="unresolved-attribute",
        line=1,
        char=len("    return model."),
    )
    assert analyzer_basic.is_false_positive(uri, diag) is True


# -- more unused-name shapes -----------------------------------------------

def test_unused_vararg_is_dropped(analyzer_basic, tmp_path):
    """``def handle(self, *args, **options)`` — the signature is imposed."""
    src = (
        "class Command:\n"
        "    def handle(self, *args, **options):\n"
        "        return 1\n"
    )
    uri = _write(tmp_path, src)
    line = src.splitlines()[1]
    for name in ("args", "options"):
        assert analyzer_basic.is_false_positive(
            uri,
            _diag(f"`{name}` is unused", code=None, line=1,
                  char=line.index(name)),
        ) is True


def test_unused_unpacked_name_is_dropped(analyzer_basic, tmp_path):
    """You can't leave an element out of a tuple unpacking."""
    src = (
        "def f(g):\n"
        "    text, actions, cost = g()\n"
        "    return text\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.is_false_positive(
        uri,
        _diag("`cost` is unused", code=None, line=1,
              char=src.splitlines()[1].index("cost")),
    ) is True


def test_unused_plain_local_is_kept(analyzer_basic, tmp_path):
    src = (
        "def f(g):\n"
        "    total = g()\n"
        "    return 1\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.is_false_positive(
        uri, _diag("`total` is unused", code=None, line=1, char=4)
    ) is False


# -- attributes on a function object ---------------------------------------

def test_attribute_on_a_decorated_function_is_dropped(analyzer_basic, tmp_path):
    """``@sync_to_async`` exposes ``.func``; ty sees the bare function."""
    uri = _write(tmp_path, "x = 1\n")
    message = (
        "Object of type `(run_id) -> Coroutine[Any, Any, Unknown]` "
        "has no attribute `func`"
    )
    assert analyzer_basic.is_false_positive(
        uri, _diag(message, code="unresolved-attribute")
    ) is True


def test_attribute_on_a_non_function_is_kept(analyzer_basic, tmp_path):
    uri = _write(tmp_path, "x = 1\n")
    assert analyzer_basic.is_false_positive(
        uri,
        _diag(
            "Object of type `Coroutine[Any, Any, Unknown]` has no attribute `func`",
            code="unresolved-attribute",
        ),
    ) is False


# -- unreachable behind a None check on a model ----------------------------

def test_unreachable_behind_a_model_none_check_is_dropped(
    analyzer_basic, tmp_path
):
    """``room = qs.first(); if room is not None: return room`` — to ty the
    model type excludes None, so the rest of the function is dead."""
    src = (
        "from myapp.models import Profile\n"
        "def f():\n"
        "    p = Profile.objects.filter(bio='x').first()\n"
        "    if p is not None:\n"
        "        return p\n"
        "    return Profile.objects.create(bio='x')\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.is_false_positive(
        uri, _diag(UNREACHABLE, code=None, line=5, char=4)
    ) is True


def test_unreachable_behind_a_plain_none_check_is_kept(
    analyzer_basic, tmp_path
):
    src = (
        "def f(g):\n"
        "    x = g()\n"
        "    if x is not None:\n"
        "        return x\n"
        "    return 0\n"
    )
    uri = _write(tmp_path, src)
    assert analyzer_basic.is_false_positive(
        uri, _diag(UNREACHABLE, code=None, line=4, char=4)
    ) is False
