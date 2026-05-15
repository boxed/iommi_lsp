"""Unit tests for DjangoAnalyzer.is_false_positive."""

from __future__ import annotations

from pathlib import Path

import pytest

from iommi_lsp.analyzers.django import DjangoAnalyzer, build_index


CORPUS = Path(__file__).parent / "corpus"


def _diag(line: int, col_start: int, col_end: int, attr: str, code: str = "unresolved-attribute"):
    return {
        "code": code,
        "message": f"Type \"…\" has no attribute \"{attr}\"",
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 1,
        "source": "ty",
    }


@pytest.fixture
def analyzer() -> DjangoAnalyzer:
    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")
    return a


def _uri_for(rel_path: str) -> str:
    return (CORPUS / rel_path).as_uri()


def test_custom_queryset_method_is_dropped(tmp_path: Path):
    """``MyQuerySet.as_manager()`` exposes custom methods on the manager —
    ty doesn't see them. We suppress on any workspace QuerySet method
    name accessed via a known model's manager."""
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "models.py").write_text(
        "from django.db import models\n"
        "\n"
        "class OrderQuerySet(models.QuerySet):\n"
        "    def active(self):\n"
        "        return self.filter(is_active=True)\n"
        "\n"
        "class Order(models.Model):\n"
        "    is_active = models.BooleanField(default=True)\n"
        "    objects = OrderQuerySet.as_manager()\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    src = (
        "from shop.models import Order\n"
        "\n"
        "def f():\n"
        "    return Order.objects.active()\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line = 3
    col = src.splitlines()[line].index("active")
    diag = _diag(line, col, col + len("active"), "active")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_unknown_method_on_manager_is_kept(tmp_path: Path):
    """A genuinely unknown method (no workspace QuerySet defines it) stays."""
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "models.py").write_text(
        "from django.db import models\n"
        "\n"
        "class OrderQuerySet(models.QuerySet):\n"
        "    def active(self):\n"
        "        return self.filter(is_active=True)\n"
        "\n"
        "class Order(models.Model):\n"
        "    is_active = models.BooleanField(default=True)\n"
        "    objects = OrderQuerySet.as_manager()\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    src = (
        "from shop.models import Order\n"
        "\n"
        "def f():\n"
        "    return Order.objects.totallybogus()\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line = 3
    col = src.splitlines()[line].index("totallybogus")
    diag = _diag(line, col, col + len("totallybogus"), "totallybogus")
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_custom_manager_subclass_methods_picked_up(tmp_path: Path):
    """Subclasses of ``models.Manager`` also surface their methods."""
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "models.py").write_text(
        "from django.db import models\n"
        "\n"
        "class OrderManager(models.Manager):\n"
        "    def recent(self):\n"
        "        return self.all()\n"
        "\n"
        "class Order(models.Model):\n"
        "    objects = OrderManager()\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    src = (
        "from shop.models import Order\n"
        "\n"
        "Order.objects.recent()\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line = 2
    col = src.splitlines()[line].index("recent")
    diag = _diag(line, col, col + len("recent"), "recent")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_get_user_model_assignment_resolves_receiver(tmp_path: Path):
    src = (
        "from django.contrib.auth import get_user_model\n"
        "\n"
        "def f():\n"
        "    UserCls = get_user_model()\n"
        "    return UserCls.objects\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_get_user_model_attribute_call(tmp_path: Path):
    """``auth.get_user_model()`` (attribute-style import)."""
    src = (
        "from django.contrib import auth\n"
        "\n"
        "def f():\n"
        "    U = auth.get_user_model()\n"
        "    return U.objects\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_m2m_through_on_class_attribute_is_dropped(tmp_path: Path):
    src = (
        "from blog.models import Tag\n"
        "\n"
        "def f():\n"
        "    return Tag.articles.through\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 3
    start = src.splitlines()[line].index("through")
    diag = _diag(line, start, start + len("through"), "through")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_m2m_through_on_instance_via_flow_is_dropped(tmp_path: Path):
    src = (
        "from blog.models import Tag\n"
        "\n"
        "def f():\n"
        "    tag = Tag.objects.get(pk=1)\n"
        "    return tag.articles.through\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 4
    start = src.splitlines()[line].index("through")
    diag = _diag(line, start, start + len("through"), "through")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_through_on_non_m2m_is_kept(tmp_path: Path):
    """``through`` on something that isn't a M2M field is a real bug."""
    src = (
        "from blog.models import Tag\n"
        "\n"
        "def f():\n"
        "    return Tag.name.through\n"   # name is a CharField
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 3
    start = src.splitlines()[line].index("through")
    diag = _diag(line, start, start + len("through"), "through")
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_objects_on_known_model_is_dropped(tmp_path: Path):
    src = "from myapp.models import User\n\ndef f():\n    return User.objects\n"
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    # Diagnostic on `objects` (line 3, "    return User.objects\n").
    # `objects` starts after `User.` at index 4 + len("return User.") = 4 + 12 = 16
    line = 3
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_unknown_model_is_kept(tmp_path: Path):
    src = "class Foo:\n    pass\n\ndef f():\n    return Foo.objects\n"
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_non_unresolved_attribute_diagnostics_are_ignored(tmp_path: Path):
    src = "from myapp.models import User\n\ndef f():\n    return User.objects\n"
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects", code="some-other-rule")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_local_flow_assignment_resolves_receiver(tmp_path: Path):
    src = (
        "from myapp.models import User\n"
        "\n"
        "def fetch():\n"
        "    user = User.objects.get(pk=1)\n"
        "    return user.pk\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4  # "    return user.pk"
    start = src.splitlines()[line].index("pk")
    diag = _diag(line, start, start + len("pk"), "pk")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_id_kept_when_explicit_pk_declared(tmp_path: Path):
    src = (
        "from myapp.models import WithExplicitPK\n"
        "\n"
        "def f():\n"
        "    return WithExplicitPK.id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    text = src.splitlines()[line]
    start = text.rindex("id")
    diag = _diag(line, start, start + 2, "id")

    # Explicit PK -> `id` is NOT auto-injected, so we should NOT suppress.
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_pk_kept_for_explicit_pk_model(tmp_path: Path):
    """`pk` is still always present even with explicit PK — must drop."""
    src = (
        "from myapp.models import WithExplicitPK\n"
        "\n"
        "def f():\n"
        "    return WithExplicitPK.pk\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    text = src.splitlines()[line]
    start = text.rindex("pk")
    diag = _diag(line, start, start + 2, "pk")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_explicit_pk_field_name_is_suppressed(tmp_path: Path):
    """Access on the model's actual PK field name must drop.

    Django's descriptor magic means ty sometimes can't see the explicit
    PK field. We look the name up off the index and suppress.
    """
    src = (
        "from myapp.models import WithExplicitPK\n"
        "\n"
        "def f():\n"
        "    return WithExplicitPK.code\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    text = src.splitlines()[line]
    start = text.rindex("code")
    diag = _diag(line, start, start + len("code"), "code")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_pk_on_annotated_param_is_dropped(tmp_path: Path):
    """``def f(u: User): u.pk`` — receiver type comes from the annotation.

    Django adds ``.pk`` to every model instance, but the flow-based
    resolver doesn't follow annotations. The annotation fallback must
    suppress this.
    """
    src = (
        "from myapp.models import User\n"
        "\n"
        "def f(u: User):\n"
        "    return u.pk\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].rindex("pk")
    diag = _diag(line, start, start + 2, "pk")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_pk_on_explicit_pk_annotated_param_is_dropped(tmp_path: Path):
    """``.pk`` still works on an instance of an explicit-PK model."""
    src = (
        "from myapp.models import WithExplicitPK\n"
        "\n"
        "def f(x: WithExplicitPK):\n"
        "    return x.pk\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].rindex("pk")
    diag = _diag(line, start, start + 2, "pk")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_pk_on_self_in_model_method_is_dropped(tmp_path: Path):
    """``self.pk`` inside a model method (explicit-PK model)."""
    src = (
        "from django.db import models\n"
        "\n"
        "class WithExplicitPK(models.Model):\n"
        "    code = models.CharField(max_length=10, primary_key=True)\n"
        "\n"
        "    def f(self):\n"
        "        return self.pk\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 6
    start = src.splitlines()[line].rindex("pk")
    diag = _diag(line, start, start + 2, "pk")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_pk_on_annotated_assignment_is_dropped(tmp_path: Path):
    """``u: User = ...; u.pk`` — annotated-assignment receiver."""
    src = (
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    u: User = get_user()\n"
        "    return u.pk\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    start = src.splitlines()[line].rindex("pk")
    diag = _diag(line, start, start + 2, "pk")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_explicit_pk_field_name_on_annotated_param_is_dropped(tmp_path: Path):
    """The actual PK field name is also suppressed on an annotated
    instance receiver — Django's descriptor magic can hide it from ty.
    """
    src = (
        "from myapp.models import WithExplicitPK\n"
        "\n"
        "def f(x: WithExplicitPK):\n"
        "    return x.code\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].rindex("code")
    diag = _diag(line, start, start + len("code"), "code")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_id_on_explicit_pk_annotated_param_is_kept(tmp_path: Path):
    """``.id`` on an explicit-PK instance is a real bug — Django does
    not inject ``id`` when the model declares ``primary_key=True``
    elsewhere."""
    src = (
        "from myapp.models import WithExplicitPK\n"
        "\n"
        "def f(x: WithExplicitPK):\n"
        "    return x.id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].rindex("id")
    diag = _diag(line, start, start + 2, "id")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_dict_comprehension_target_resolves_receiver(tmp_path: Path):
    """``{u.id: u for u in User.objects.all()}`` — comprehension target.

    The comprehension binds ``u`` to a ``User`` instance, so ``u.id``
    (an implicit-PK attribute) must be suppressed.
    """
    src = (
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    return {u.id: u for u in User.objects.all()}\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    text = src.splitlines()[line]
    # The first "id" in the line — i.e., the `u.id` key.
    start = text.index(".id") + 1
    diag = _diag(line, start, start + 2, "id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_for_loop_target_resolves_receiver(tmp_path: Path):
    """``for u in User.objects.filter(...)`` — for-loop target binding."""
    src = (
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    for u in User.objects.filter(email='x'):\n"
        "        print(u.id)\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    text = src.splitlines()[line]
    start = text.index(".id") + 1
    diag = _diag(line, start, start + 2, "id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_reverse_relation_is_dropped(tmp_path: Path):
    src = (
        "from blog.models import Author\n"
        "\n"
        "def f():\n"
        "    a = Author.objects.first()\n"
        "    return a.articles\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 4
    start = src.splitlines()[line].index("articles")
    diag = _diag(line, start, start + len("articles"), "articles")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_default_reverse_set_is_dropped(tmp_path: Path):
    src = (
        "from blog.models import Article\n"
        "\n"
        "def f():\n"
        "    a = Article.objects.first()\n"
        "    return a.comment_set\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 4
    start = src.splitlines()[line].index("comment_set")
    diag = _diag(line, start, start + len("comment_set"), "comment_set")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_reverse_relation_on_annotated_param_is_dropped(tmp_path: Path):
    """``def f(a: Author): a.articles`` — related_name via annotation."""
    src = (
        "from blog.models import Author\n"
        "\n"
        "def f(a: Author):\n"
        "    return a.articles\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 3
    start = src.splitlines()[line].index("articles")
    diag = _diag(line, start, start + len("articles"), "articles")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_default_reverse_set_on_annotated_param_is_dropped(tmp_path: Path):
    """``def f(a: Article): a.comment_set`` — default ``*_set`` via annotation."""
    src = (
        "from blog.models import Article\n"
        "\n"
        "def f(a: Article):\n"
        "    return a.comment_set\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 3
    start = src.splitlines()[line].index("comment_set")
    diag = _diag(line, start, start + len("comment_set"), "comment_set")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_reverse_relation_on_self_in_model_method_is_dropped(tmp_path: Path):
    """``def method(self): self.comment_set`` inside the target model class."""
    src = (
        "from django.db import models\n"
        "\n"
        "class Article(models.Model):\n"
        "    title = models.CharField(max_length=200)\n"
        "\n"
        "    def f(self):\n"
        "        return self.comment_set\n"
        "\n"
        "class Comment(models.Model):\n"
        "    article = models.ForeignKey(Article, on_delete=models.CASCADE)\n"
    )
    f = tmp_path / "blog" / "models.py"
    f.parent.mkdir()
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    line = 6
    start = src.splitlines()[line].index("comment_set")
    diag = _diag(line, start, start + len("comment_set"), "comment_set")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_reverse_relation_on_annotated_assignment_is_dropped(tmp_path: Path):
    """``a: Article = ...; a.comment_set`` — annotated assignment receiver."""
    src = (
        "from blog.models import Article\n"
        "\n"
        "def f():\n"
        "    a: Article = get_article()\n"
        "    return a.comment_set\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 4
    start = src.splitlines()[line].index("comment_set")
    diag = _diag(line, start, start + len("comment_set"), "comment_set")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_m2m_reverse_relation_on_annotated_param_is_dropped(tmp_path: Path):
    """``def f(a: Article): a.tags`` — M2M reverse with related_name."""
    src = (
        "from blog.models import Article\n"
        "\n"
        "def f(a: Article):\n"
        "    return a.tags\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 3
    start = src.splitlines()[line].index("tags")
    diag = _diag(line, start, start + len("tags"), "tags")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_unknown_reverse_on_annotated_param_is_kept(tmp_path: Path):
    """``def f(a: Article): a.bogus_set`` — not a real reverse, keep diag."""
    src = (
        "from blog.models import Article\n"
        "\n"
        "def f(a: Article):\n"
        "    return a.bogus_set\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 3
    start = src.splitlines()[line].index("bogus_set")
    diag = _diag(line, start, start + len("bogus_set"), "bogus_set")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_hidden_reverse_with_plus_is_kept(tmp_path: Path):
    """``related_name='+'`` disables the reverse — ``a.hiddenlink_set`` is a real bug."""
    src = (
        "from blog.models import Article\n"
        "\n"
        "def f(a: Article):\n"
        "    return a.hiddenlink_set\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 3
    start = src.splitlines()[line].index("hiddenlink_set")
    diag = _diag(line, start, start + len("hiddenlink_set"), "hiddenlink_set")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_self_referential_related_name_reverse_is_dropped(tmp_path: Path):
    """``ForeignKey(Foo, related_name='foos')`` — the reverse ``x.foos`` on a
    ``Foo`` instance must not warn from ty. Covers the case where the
    related_name lives on a sibling model pointing back at ``Foo``.
    """
    models_src = (
        "from django.db import models\n"
        "\n"
        "class Foo(models.Model):\n"
        "    name = models.CharField(max_length=200)\n"
        "\n"
        "class FooChild(models.Model):\n"
        "    parent = models.ForeignKey(\n"
        "        Foo, on_delete=models.CASCADE, related_name='foos',\n"
        "    )\n"
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "models.py").write_text(models_src)

    user_src = (
        "from app.models import Foo\n"
        "\n"
        "def f(x: Foo):\n"
        "    return x.foos\n"
    )
    u = tmp_path / "u.py"
    u.write_text(user_src)

    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    line = 3
    start = user_src.splitlines()[line].index("foos")
    diag = _diag(line, start, start + len("foos"), "foos")

    assert a.is_false_positive(u.as_uri(), diag) is True


def test_resolve_definition_jumps_to_related_name_fk(tmp_path: Path):
    """``x.foos`` should resolve to the ``parent = ForeignKey(Foo,
    related_name='foos')`` declaration on the source model."""
    models_src = (
        "from django.db import models\n"
        "\n"
        "class Foo(models.Model):\n"
        "    name = models.CharField(max_length=200)\n"
        "\n"
        "class FooChild(models.Model):\n"
        "    parent = models.ForeignKey(\n"
        "        Foo, on_delete=models.CASCADE, related_name='foos',\n"
        "    )\n"
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    models_path = tmp_path / "app" / "models.py"
    models_path.write_text(models_src)

    user_src = (
        "from app.models import Foo\n"
        "\n"
        "def f(x: Foo):\n"
        "    return x.foos\n"
    )
    u = tmp_path / "u.py"
    u.write_text(user_src)

    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    # Cursor sits inside the ``foos`` token on the ``return x.foos`` line.
    line = 3
    foos_col = user_src.splitlines()[line].index("foos")
    loc = a.resolve_definition(
        u.as_uri(), {"line": line, "character": foos_col + 1},
    )
    assert loc is not None
    assert loc["uri"] == models_path.as_uri()

    # The range points at the ``parent`` token on its declaration line.
    models_lines = models_src.splitlines()
    # FieldInfo records the LHS-name token's location; for the wrapped
    # ForeignKey call the assignment statement begins on the ``parent =``
    # line, and ast.lineno/col_offset for the target Name node match that.
    expected_line = next(
        i for i, ln in enumerate(models_lines) if ln.lstrip().startswith("parent =")
    )
    expected_col = models_lines[expected_line].index("parent")
    assert loc["range"]["start"] == {"line": expected_line, "character": expected_col}
    assert loc["range"]["end"] == {
        "line": expected_line,
        "character": expected_col + len("parent"),
    }


def test_resolve_definition_returns_none_for_non_reverse_attr(tmp_path: Path):
    """``x.unrelated`` shouldn't get hijacked — let ty answer."""
    models_src = (
        "from django.db import models\n"
        "\n"
        "class Foo(models.Model):\n"
        "    name = models.CharField(max_length=200)\n"
        "\n"
        "class FooChild(models.Model):\n"
        "    parent = models.ForeignKey(\n"
        "        Foo, on_delete=models.CASCADE, related_name='foos',\n"
        "    )\n"
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "models.py").write_text(models_src)

    user_src = (
        "from app.models import Foo\n"
        "\n"
        "def f(x: Foo):\n"
        "    return x.name\n"
    )
    u = tmp_path / "u.py"
    u.write_text(user_src)

    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    line = 3
    col = user_src.splitlines()[line].index("name") + 1
    loc = a.resolve_definition(u.as_uri(), {"line": line, "character": col})
    assert loc is None


def test_fk_id_accessor_is_dropped(tmp_path: Path):
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f():\n"
        "    return Profile.user_id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("user_id")
    diag = _diag(line, start, start + len("user_id"), "user_id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_fk_id_accessor_on_annotated_param_is_dropped(tmp_path: Path):
    """`def f(p: Profile): p.user_id` — ty knows p's type from the
    annotation; the flow-based resolver doesn't, so verify the
    annotation fallback kicks in for fk_id."""
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f(p: Profile):\n"
        "    return p.user_id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("user_id")
    diag = _diag(line, start, start + len("user_id"), "user_id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_fk_id_accessor_on_self_in_model_method_is_dropped(tmp_path: Path):
    """`def method(self): self.user_id` inside the model itself."""
    src = (
        "from django.db import models\n"
        "from myapp.models import User\n"
        "\n"
        "class Profile(models.Model):\n"
        "    user = models.ForeignKey(User, on_delete=models.CASCADE)\n"
        "\n"
        "    def f(self):\n"
        "        return self.user_id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 7
    start = src.splitlines()[line].index("user_id")
    diag = _diag(line, start, start + len("user_id"), "user_id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_fk_id_accessor_on_optional_annotation_is_dropped(tmp_path: Path):
    """`def f(p: Profile | None): p.user_id` — unwrap the union."""
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f(p: Profile | None):\n"
        "    return p.user_id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("user_id")
    diag = _diag(line, start, start + len("user_id"), "user_id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_fk_id_accessor_on_annotated_assignment_is_dropped(tmp_path: Path):
    """`p: Profile = get_profile(); p.user_id` — annotated assignment."""
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f():\n"
        "    p: Profile = get_profile()\n"
        "    return p.user_id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    start = src.splitlines()[line].index("user_id")
    diag = _diag(line, start, start + len("user_id"), "user_id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_non_fk_id_attribute_on_annotated_param_is_kept(tmp_path: Path):
    """The annotation fallback is narrow: it only suppresses fk_id, not
    e.g. `p.objects` (which would be a real bug on an instance)."""
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f(p: Profile):\n"
        "    return p.objects\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("objects")
    diag = _diag(line, start, start + len("objects"), "objects")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_fk_id_accessor_on_chained_queryset_first_is_dropped(tmp_path: Path):
    """`p = Profile.objects.filter(...).first(); p.user_id` — instance
    bound from a chained queryset call. Common real-world shape; the
    flow resolver must walk through queryset-returning methods to the
    manager."""
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f():\n"
        "    p = Profile.objects.filter(bio='x').first()\n"
        "    return p.user_id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    start = src.splitlines()[line].index("user_id")
    diag = _diag(line, start, start + len("user_id"), "user_id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_fk_id_accessor_on_deeply_chained_queryset_is_dropped(tmp_path: Path):
    """Several queryset-returning methods chained before the terminal
    instance-returning call."""
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f():\n"
        "    p = Profile.objects.filter(bio='x').exclude(bio='y').order_by('bio').first()\n"
        "    return p.user_id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    start = src.splitlines()[line].index("user_id")
    diag = _diag(line, start, start + len("user_id"), "user_id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_unknown_fk_id_on_chained_queryset_is_kept(tmp_path: Path):
    """Chain-resolved receiver still rejects a genuinely-unknown
    ``<name>_id`` accessor (no ``bogus`` FK on Profile)."""
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f():\n"
        "    p = Profile.objects.filter(bio='x').first()\n"
        "    return p.bogus_id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    start = src.splitlines()[line].index("bogus_id")
    diag = _diag(line, start, start + len("bogus_id"), "bogus_id")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_unknown_fk_id_on_annotated_param_is_kept(tmp_path: Path):
    """`p.bogus_id` where `bogus` is not an FK on Profile — real bug."""
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f(p: Profile):\n"
        "    return p.bogus_id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("bogus_id")
    diag = _diag(line, start, start + len("bogus_id"), "bogus_id")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_unrelated_attribute_is_kept(tmp_path: Path):
    """Genuine unresolved attribute on a model must NOT be suppressed."""
    src = (
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    return User.objects_typo\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("objects_typo")
    diag = _diag(line, start, start + len("objects_typo"), "objects_typo")

    assert a.is_false_positive(f.as_uri(), diag) is False


def _unused_diag(line: int, col_start: int, col_end: int, name: str = "request"):
    """Mirror ty's actual ``\\`x\\` is unused`` hint shape (no ``code``)."""
    return {
        "message": f"`{name}` is unused",
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 4,
        "source": "ty",
        "tags": [1],
    }


def test_unused_request_first_param_is_dropped(tmp_path: Path):
    """`def view(request): ...` — request unused, drop ty's hint."""
    src = "def my_view(request):\n    return None\n"
    f = tmp_path / "v.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 0
    start = src.splitlines()[line].index("request")
    diag = _unused_diag(line, start, start + len("request"))

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_unused_request_on_method_is_dropped(tmp_path: Path):
    """CBV-style `def get(self, request, ...)` — `self` skipped, request is first."""
    src = (
        "class V:\n"
        "    def get(self, request):\n"
        "        return None\n"
    )
    f = tmp_path / "v.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 1
    start = src.splitlines()[line].index("request")
    diag = _unused_diag(line, start, start + len("request"))

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_unused_request_not_first_param_is_kept(tmp_path: Path):
    """`def f(x, request)` — request isn't the first non-self/cls arg, keep the hint."""
    src = "def helper(x, request):\n    return x\n"
    f = tmp_path / "v.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 0
    start = src.splitlines()[line].index("request")
    diag = _unused_diag(line, start, start + len("request"))

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_unused_non_request_param_is_kept(tmp_path: Path):
    """Only `request` gets the exception — other unused params still flag."""
    src = "def helper(payload):\n    return None\n"
    f = tmp_path / "v.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 0
    start = src.splitlines()[line].index("payload")
    diag = _unused_diag(line, start, start + len("payload"), name="payload")

    assert a.is_false_positive(f.as_uri(), diag) is False


def test_unused_request_local_variable_is_kept(tmp_path: Path):
    """An unused *local variable* named `request` still flags — only the
    function parameter position is whitelisted."""
    src = (
        "def handler():\n"
        "    request = None\n"
        "    return None\n"
    )
    f = tmp_path / "v.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 1
    start = src.splitlines()[line].index("request")
    diag = _unused_diag(line, start, start + len("request"))

    assert a.is_false_positive(f.as_uri(), diag) is False


def _invalid_enum_diag(line: int, col_start: int, col_end: int, name: str) -> dict:
    """Mirror ty's ``invalid-assignment`` shape for an Enum tuple member."""
    return {
        "code": "invalid-assignment",
        "message": f"Enum member `{name}` is incompatible with `__new__`",
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 1,
        "source": "ty",
    }


@pytest.mark.parametrize("base", ["models.IntegerChoices", "models.TextChoices"])
def test_choices_enum_member_invalid_assignment_is_dropped(tmp_path: Path, base: str):
    src = (
        "from django.db import models\n"
        "\n"
        f"class MyChoices({base}):\n"
        "    GOOD = 1, \"I like this\"\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("GOOD")
    diag = _invalid_enum_diag(line, start, start + len("GOOD"), "GOOD")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_choices_enum_member_bare_import(tmp_path: Path):
    """``from django.db.models import IntegerChoices`` — bare Name base."""
    src = (
        "from django.db.models import IntegerChoices\n"
        "\n"
        "class MyChoices(IntegerChoices):\n"
        "    GOOD = 1, \"I like this\"\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("GOOD")
    diag = _invalid_enum_diag(line, start, start + len("GOOD"), "GOOD")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_invalid_assignment_outside_choices_class_is_kept(tmp_path: Path):
    """A plain class that happens to hit ``invalid-assignment`` — keep it."""
    src = (
        "class Plain:\n"
        "    x: int = \"oops\"\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 1
    start = src.splitlines()[line].index("x")
    diag = _invalid_enum_diag(line, start, start + 1, "x")
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_invalid_assignment_non_enum_message_is_kept(tmp_path: Path):
    """``invalid-assignment`` with a non-Enum message — keep it even inside Choices.

    We only suppress ty's Enum-specific complaint; other ``invalid-assignment``
    errors on a Choices class (e.g. a method body bug) are still real.
    """
    src = (
        "from django.db import models\n"
        "\n"
        "class MyChoices(models.IntegerChoices):\n"
        "    GOOD = 1, \"I like this\"\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    start = src.splitlines()[line].index("GOOD")
    diag = {
        "code": "invalid-assignment",
        "message": "Type `str` is not assignable to `int`",
        "range": {
            "start": {"line": line, "character": start},
            "end": {"line": line, "character": start + len("GOOD")},
        },
        "severity": 1,
        "source": "ty",
    }
    assert a.is_false_positive(f.as_uri(), diag) is False
