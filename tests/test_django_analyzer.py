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


def test_shared_model_name_disambiguated_by_import(tmp_path: Path):
    """Two apps define a `Service` model, so the bare-name lookup ties.
    The using file's import (`from core.models import Service`) pins it to
    one model, so a reverse-relation access on an instance is recognised
    and the false unresolved-attribute is suppressed."""
    for app in ("core", "prospects"):
        (tmp_path / app).mkdir()
        (tmp_path / app / "__init__.py").write_text("")
    (tmp_path / "prospects" / "models.py").write_text(
        "from django.db import models\n"
        "class Service(models.Model):\n"
        "    name = models.CharField(max_length=50)\n"
    )
    (tmp_path / "core" / "models.py").write_text(
        "from django.db import models\n"
        "class Service(models.Model):\n"
        "    name = models.CharField(max_length=50)\n"
        "class ActionChain(models.Model):\n"
        "    service = models.ForeignKey(Service, on_delete=models.CASCADE,\n"
        "                                related_name='action_chains')\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    src = (
        "from core.models import Service\n"
        "\n"
        "def f(pk):\n"
        "    service = Service.objects.get(pk=pk)\n"
        "    return list(service.action_chains.all())\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line = 4
    col = src.splitlines()[line].index("action_chains")
    diag = _diag(line, col, col + len("action_chains"), "action_chains")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_shared_model_name_genuine_typo_still_kept(tmp_path: Path):
    """Same shared-name setup, but a genuine typo on the disambiguated
    model must still surface (no over-suppression)."""
    for app in ("core", "prospects"):
        (tmp_path / app).mkdir()
        (tmp_path / app / "__init__.py").write_text("")
    (tmp_path / "prospects" / "models.py").write_text(
        "from django.db import models\n"
        "class Service(models.Model):\n"
        "    name = models.CharField(max_length=50)\n"
    )
    (tmp_path / "core" / "models.py").write_text(
        "from django.db import models\n"
        "class Service(models.Model):\n"
        "    name = models.CharField(max_length=50)\n"
        "class ActionChain(models.Model):\n"
        "    service = models.ForeignKey(Service, on_delete=models.CASCADE,\n"
        "                                related_name='action_chains')\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    src = (
        "from core.models import Service\n"
        "\n"
        "def f(pk):\n"
        "    service = Service.objects.get(pk=pk)\n"
        "    return service.action_chainz\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)
    line = 4
    col = src.splitlines()[line].index("action_chainz")
    diag = _diag(line, col, col + len("action_chainz"), "action_chainz")
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_shared_model_name_disambiguated_by_relative_import(tmp_path: Path):
    """Same shared-name setup, but the using file reaches the model via a
    relative import (``from .models import Service``). The package context
    of the importing module must resolve it to the right app's model."""
    for app in ("core", "prospects"):
        (tmp_path / app).mkdir()
        (tmp_path / app / "__init__.py").write_text("")
    (tmp_path / "prospects" / "models.py").write_text(
        "from django.db import models\n"
        "class Service(models.Model):\n"
        "    name = models.CharField(max_length=50)\n"
    )
    (tmp_path / "core" / "models.py").write_text(
        "from django.db import models\n"
        "class Service(models.Model):\n"
        "    name = models.CharField(max_length=50)\n"
        "class ActionChain(models.Model):\n"
        "    service = models.ForeignKey(Service, on_delete=models.CASCADE,\n"
        "                                related_name='action_chains')\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    # `core/views.py` (module core.views) reaches Service via `.models`.
    src = (
        "from .models import Service\n"
        "\n"
        "def f(pk):\n"
        "    service = Service.objects.get(pk=pk)\n"
        "    return list(service.action_chains.all())\n"
    )
    f = tmp_path / "core" / "views.py"
    f.write_text(src)
    line = 4
    col = src.splitlines()[line].index("action_chains")
    diag = _diag(line, col, col + len("action_chains"), "action_chains")
    assert a.is_false_positive(f.as_uri(), diag) is True


def _self_diag(line: int, col_start: int, col_end: int, attr: str, method: str):
    """A ty ``unresolved-attribute`` whose receiver is a ``Self@<method>``
    type — what ty emits for ``self`` and Self-typed locals inside a method."""
    return {
        "code": "unresolved-attribute",
        "message": f"Object of type `Self@{method}` has no attribute `{attr}`",
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 1,
        "source": "ty",
    }


_MIXIN_CORPUS = (
    "from django.db import models\n"
    "import copy\n"
    "class DuplicateSupport:\n"
    "    def duplicate(self):\n"
    "        result = copy.deepcopy(self)\n"
    "        result.based_on_id = self.pk\n"   # fk_id + universal magic
    "        if result.master is not None:\n"  # declared field on all users
    "            return result.children\n"     # not on any user -> kept
    "class Object(models.Model, DuplicateSupport):\n"
    "    master = models.ForeignKey('Master', on_delete=models.CASCADE)\n"
    "    based_on = models.ForeignKey('self', null=True, on_delete=models.SET_NULL)\n"
    "class Service(models.Model, DuplicateSupport):\n"
    "    master = models.ForeignKey('Master', on_delete=models.CASCADE)\n"
    "    based_on = models.ForeignKey('self', null=True, on_delete=models.SET_NULL)\n"
    "class Master(models.Model):\n"
    "    name = models.CharField(max_length=50)\n"
)


def test_self_typed_receiver_in_mixin_suppressed(tmp_path: Path):
    """``self``/Self-typed locals inside a non-model mixin resolve to the
    concrete models that inherit it. Universal magic (``pk``), shared
    declared fields (``master``), and shared ``<fk>_id`` accessors
    (``based_on_id``) are suppressed; this covers both ``self`` and
    ``result = deepcopy(self)`` since ty types both as ``Self@duplicate``."""
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "models.py").write_text(_MIXIN_CORPUS)
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)
    f = tmp_path / "shop" / "models.py"
    src = f.read_text().splitlines()

    def check(lineno_1, recv_attr, method):
        line = src[lineno_1 - 1]
        attr = recv_attr.split(".", 1)[1]
        col = line.index(recv_attr) + recv_attr.index(".") + 1
        return a.is_false_positive(
            f.as_uri(), _self_diag(lineno_1 - 1, col, col + len(attr), attr, method)
        )

    assert check(6, "self.pk", "duplicate") is True            # universal magic
    assert check(6, "result.based_on_id", "duplicate") is True  # shared fk_id
    assert check(7, "result.master", "duplicate") is True       # shared field


def test_self_typed_receiver_in_mixin_typo_kept(tmp_path: Path):
    """An attribute that doesn't resolve on the mixin's users still surfaces."""
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "models.py").write_text(_MIXIN_CORPUS)
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)
    f = tmp_path / "shop" / "models.py"
    src = f.read_text().splitlines()
    line = src[7]
    col = line.index("result.children") + len("result")
    diag = _self_diag(7, col, col + len("children"), "children", "duplicate")
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_self_typed_field_on_only_some_users_kept(tmp_path: Path):
    """A field present on one mixin user but not another must NOT be
    suppressed — it's a latent bug for the user that lacks it."""
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "models.py").write_text(
        "from django.db import models\n"
        "class Mixin:\n"
        "    def go(self):\n"
        "        return self.only_on_a\n"
        "class A(models.Model, Mixin):\n"
        "    only_on_a = models.CharField(max_length=5)\n"
        "class B(models.Model, Mixin):\n"
        "    name = models.CharField(max_length=5)\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)
    f = tmp_path / "shop" / "models.py"
    src = f.read_text().splitlines()
    line = src[3]
    col = line.index(".only_on_a") + 1
    diag = _self_diag(3, col, col + len("only_on_a"), "only_on_a", "go")
    assert a.is_false_positive(f.as_uri(), diag) is False


def _union_diag(line: int, col_start: int, col_end: int, attr: str, missing_on: str, union: str):
    """A ty ``unresolved-attribute`` whose message reports against one arm
    of a union type (``Attribute `x` is not defined on `T` in union `…`)."""
    return {
        "code": "unresolved-attribute",
        "message": f"Attribute `{attr}` is not defined on `{missing_on}` in union `{union}`",
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 1,
        "source": "ty",
    }


def test_union_message_reverse_attr_suppressed(tmp_path: Path):
    """ty reports against one arm of a partially-inferred union
    (``Unknown | ActionChain``). The AST can't resolve the receiver (it's a
    list-comprehension rebinding), so resolution falls to the message — the
    flagged arm ``ActionChain`` does carry the reverse accessor, so suppress."""
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "models.py").write_text(
        "from django.db import models\n"
        "class ActionChain(models.Model):\n"
        "    name = models.CharField(max_length=50)\n"
        "class Action(models.Model):\n"
        "    action_chain = models.ForeignKey(ActionChain, on_delete=models.CASCADE,\n"
        "                                      related_name='actions')\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    src = (
        "from shop.models import ActionChain\n"
        "def f(chains):\n"
        "    chains = [c for c in chains]\n"
        "    for ac in chains:\n"
        "        return ac.actions.all()\n"
    )
    f = tmp_path / "shop" / "u.py"
    f.write_text(src)
    line = 4
    col = src.splitlines()[line].index(".actions") + 1
    diag = _union_diag(
        line, col, col + len("actions"), "actions",
        "ActionChain", "Unknown | ActionChain",
    )
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_union_message_genuine_typo_kept(tmp_path: Path):
    """Same union shape, but the attribute genuinely doesn't exist on the
    flagged arm — the warning must survive."""
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "models.py").write_text(
        "from django.db import models\n"
        "class ActionChain(models.Model):\n"
        "    name = models.CharField(max_length=50)\n"
        "class Action(models.Model):\n"
        "    action_chain = models.ForeignKey(ActionChain, on_delete=models.CASCADE,\n"
        "                                      related_name='actions')\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    src = (
        "from shop.models import ActionChain\n"
        "def f(chains):\n"
        "    chains = [c for c in chains]\n"
        "    for ac in chains:\n"
        "        return ac.bogus_xyz\n"
    )
    f = tmp_path / "shop" / "u.py"
    f.write_text(src)
    line = 4
    col = src.splitlines()[line].index(".bogus_xyz") + 1
    diag = _union_diag(
        line, col, col + len("bogus_xyz"), "bogus_xyz",
        "ActionChain", "Unknown | ActionChain",
    )
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_self_reverse_attr_in_model_method_with_shared_name(tmp_path: Path):
    """``self.<reverse>`` inside a model method must resolve even when the
    model's simple name is shared across apps. ty reports the receiver as
    ``Self@<method>``; resolution goes through the enclosing class, whose
    name is disambiguated by the file's own module."""
    for app in ("core", "prospects"):
        (tmp_path / app).mkdir()
        (tmp_path / app / "__init__.py").write_text("")
    # A second `Service` elsewhere makes the bare name ambiguous.
    (tmp_path / "core" / "models.py").write_text(
        "from django.db import models\n"
        "class Service(models.Model):\n"
        "    name = models.CharField(max_length=50)\n"
    )
    (tmp_path / "prospects" / "models.py").write_text(
        "from django.db import models\n"
        "class Service(models.Model):\n"
        "    name = models.CharField(max_length=50)\n"
        "    def zero_out(self):\n"
        "        return self.moments.all()\n"
        "class Moment(models.Model):\n"
        "    service = models.ForeignKey(Service, on_delete=models.CASCADE,\n"
        "                                related_name='moments')\n"
    )
    a = DjangoAnalyzer(workspace_root=tmp_path)
    a.django_index = build_index(tmp_path)

    f = tmp_path / "prospects" / "models.py"
    src = f.read_text()
    line = 4  # `        return self.moments.all()`
    col = src.splitlines()[line].index(".moments") + 1
    diag = _diag(line, col, col + len("moments"), "moments")
    diag["message"] = "Object of type `Self@zero_out` has no attribute `moments`"
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


def test_message_based_fallback_suppresses_pk_on_unannotated_param(tmp_path: Path):
    """ty-semantic infers a model type for receivers we can't reach from
    the AST (e.g. an ``@decode_path``-wrapped view's ``project=None``
    kwarg). The receiver is a bare ``Name`` with no annotation and no
    in-scope assignment, so AST-based resolution returns nothing — but
    ty's own message names the type: ``Object of type `User` has no
    attribute `id```. The fallback parses that name and re-runs the
    magic-attr check against the index.
    """
    src = (
        "def view(request, *, user=None):\n"
        "    return user.id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 1
    start = src.splitlines()[line].index("user.id")
    diag = {
        "code": "unresolved-attribute",
        "message": "Object of type `User` has no attribute `id`",
        "range": {
            "start": {"line": line, "character": start},
            "end": {"line": line, "character": start + len("user.id")},
        },
        "severity": 1,
        "source": "ty",
    }
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_message_based_fallback_keeps_diagnostic_for_explicit_pk_model(tmp_path: Path):
    """``.id`` on an explicit-PK model is a real bug — the fallback must
    still respect that by running ``_attr_is_magic``, which returns False
    here.
    """
    src = (
        "def view(request, *, x=None):\n"
        "    return x.id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 1
    start = src.splitlines()[line].index("x.id")
    diag = {
        "code": "unresolved-attribute",
        "message": "Object of type `WithExplicitPK` has no attribute `id`",
        "range": {
            "start": {"line": line, "character": start},
            "end": {"line": line, "character": start + len("x.id")},
        },
        "severity": 1,
        "source": "ty",
    }
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_message_based_fallback_keeps_diagnostic_for_unknown_type(tmp_path: Path):
    """A type name that isn't in the index falls through — we'd rather
    leak noise than suppress a real attribute error on a non-model class.
    """
    src = (
        "def view(*, x=None):\n"
        "    return x.id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 1
    start = src.splitlines()[line].index("x.id")
    diag = {
        "code": "unresolved-attribute",
        "message": "Object of type `NotAModel` has no attribute `id`",
        "range": {
            "start": {"line": line, "character": start},
            "end": {"line": line, "character": start + len("x.id")},
        },
        "severity": 1,
        "source": "ty",
    }
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_message_based_fallback_handles_legacy_quoted_format(tmp_path: Path):
    """Older ty variants spelled the type ``Type "User" has no attribute
    "id"``. Same receiver shape, same expected outcome.
    """
    src = (
        "def view(*, user=None):\n"
        "    return user.id\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 1
    start = src.splitlines()[line].index("user.id")
    diag = {
        "code": "unresolved-attribute",
        "message": 'Type "User" has no attribute "id"',
        "range": {
            "start": {"line": line, "character": start},
            "end": {"line": line, "character": start + len("user.id")},
        },
        "severity": 1,
        "source": "ty",
    }
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


def test_comprehension_over_reassigned_queryset_resolves_receiver(tmp_path: Path):
    """``authors = authors.filter(...)`` then ``for a in authors`` in a
    comprehension. The receiver ``a`` must still resolve to ``Author`` so
    reverse-FK accessors like ``a.articles`` are recognised. Previously
    the back-walk for ``authors`` picked up the self-referential
    ``authors = authors.filter(...)`` value and hit the depth limit.
    """
    src = (
        "from blog.models import Author\n"
        "\n"
        "def f(flag):\n"
        "    authors = Author.objects.filter(name='a')\n"
        "    if flag:\n"
        "        authors = authors.filter(name='b')\n"
        "    return [a.articles for a in authors]\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 6
    text = src.splitlines()[line]
    start = text.index(".articles") + 1
    diag = _diag(line, start, start + len("articles"), "articles")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_reverse_on_multi_branch_reassigned_queryset_in_nested_comp(tmp_path: Path):
    """The full user-reported shape: ``if/elif`` reassignments of the
    queryset, then a multi-generator comprehension whose first generator
    binds ``a`` and whose second generator uses ``a.articles`` (a reverse
    FK). Two reassignments push the back-walk past the original
    ``_depth < 4`` cap in ``_infer_iter_yields_model``.
    """
    src = (
        "from blog.models import Author\n"
        "\n"
        "def f(flavour):\n"
        "    authors = Author.objects.filter(name='a')\n"
        "    if flavour == 'x':\n"
        "        authors = authors.filter(name='b')\n"
        "    elif flavour == 'y':\n"
        "        authors = authors.filter(name='c')\n"
        "    return [\n"
        "        x\n"
        "        for a in authors\n"
        "        for x in a.articles.all()\n"
        "    ]\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 11  # `        for x in a.articles.all()`
    text = src.splitlines()[line]
    start = text.index(".articles") + 1
    diag = _diag(line, start, start + len("articles"), "articles")

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


def test_default_reverse_set_on_unannotated_param_via_message(tmp_path: Path):
    """``def f(project): project.job_set`` — receiver has no annotation, so
    we can only know the type from ty's diagnostic message
    (``Object of type `Project` has no attribute `job_set```). The
    message-based fallback resolves ``Project`` to the model and drops
    the diagnostic the same as the annotated/flow cases above.
    """
    src = (
        "def f(a):\n"
        "    return a.comment_set\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "related_names")
    a.django_index = build_index(CORPUS / "related_names")

    line = 1
    start = src.splitlines()[line].index("a.comment_set")
    diag = {
        "code": "unresolved-attribute",
        "message": "Object of type `Article` has no attribute `comment_set`",
        "range": {
            "start": {"line": line, "character": start},
            "end": {"line": line, "character": start + len("a.comment_set")},
        },
        "severity": 1,
        "source": "ty",
    }
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


def test_fk_id_accessor_in_for_loop_over_named_queryset_is_dropped(tmp_path: Path):
    """Real-world shape: a chained queryset is bound to a name, then a
    ``for`` loop iterates that name. ``dp.user_id`` in the loop body
    must resolve through both layers — first the loop target binding,
    then the queryset variable's binding."""
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f():\n"
        "    profiles = (\n"
        "        Profile.objects.filter(bio='x')\n"
        "        .filter(bio__isnull=False)\n"
        "        .select_related('user')\n"
        "    )\n"
        "    for dp in profiles:\n"
        "        print(dp.user_id)\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 9
    start = src.splitlines()[line].index("user_id")
    diag = _diag(line, start, start + len("user_id"), "user_id")

    assert a.is_false_positive(f.as_uri(), diag) is True


def test_fk_id_accessor_in_comprehension_over_named_queryset_is_dropped(tmp_path: Path):
    """Same as the for-loop case but in a comprehension's iter."""
    src = (
        "from myapp.models import Profile\n"
        "\n"
        "def f():\n"
        "    profiles = Profile.objects.filter(bio='x').select_related('user')\n"
        "    return [dp.user_id for dp in profiles]\n"
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


def test_unused_request_on_decorated_module_view_is_dropped(tmp_path: Path):
    """Real-world shape: a decorated module-level view ``@x.allow / def v(request):``.

    Regression for the case actually seen in the wild — ty reports the hint
    at the parameter even when the function carries a decorator. The
    decorator must not throw off the first-param match (the param's own
    lineno/col is what counts, not the function's).
    """
    src = (
        "import permissions\n"
        "\n"
        "\n"
        "@permissions.allow\n"
        "def blank_internal(request):\n"
        "    return None\n"
    )
    f = tmp_path / "views.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    start = src.splitlines()[line].index("request")
    diag = _unused_diag(line, start, start + len("request"))

    assert a.is_false_positive(f.as_uri(), diag) is True


@pytest.mark.asyncio
async def test_unused_request_dropped_end_to_end_through_interceptor(tmp_path: Path):
    """End-to-end: the hint is stripped from the publishDiagnostics frame the
    editor sees, not merely flagged by ``is_false_positive`` in isolation.

    This guards the whole path — interceptor → analyzer → first-param check —
    so a regression anywhere in the chain (not just the predicate) trips it.
    """
    import json

    from iommi_lsp.interceptor import DiagnosticInterceptor

    src = (
        "@permissions.allow\n"
        "def blank_internal(request):\n"
        "    return None\n"
    )
    f = tmp_path / "views.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")
    interceptor = DiagnosticInterceptor(analyzers=[a])

    line = 1
    start = src.splitlines()[line].index("request")
    payload = {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": f.as_uri(),
            "diagnostics": [_unused_diag(line, start, start + len("request"))],
        },
    }
    out = await interceptor(json.dumps(payload).encode("utf-8"))
    assert out is not None
    forwarded = json.loads(out)
    assert forwarded["params"]["diagnostics"] == []


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


# -- annotate() alias suppression --------------------------------------------


def test_annotate_alias_on_for_loop_instance_is_dropped(tmp_path: Path):
    """``for u in User.objects.annotate(n=…): u.n`` — ty doesn't know
    ``n`` is added at runtime; we walk the queryset chain bound by the
    ``for`` and recognise the alias."""
    src = (
        "from django.db.models import Count\n"
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    for u in User.objects.annotate(n=Count('id')):\n"
        "        print(u.n)\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 5
    col = src.splitlines()[line].index("u.n") + 2
    diag = _diag(line, col, col + 1, "n")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_annotate_alias_via_qs_variable_is_dropped(tmp_path: Path):
    """Cross-statement: ``qs = M.objects.annotate(n=…); for u in qs: u.n``
    — the alias resolver walks back through the ``qs`` binding."""
    src = (
        "from django.db.models import Count\n"
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    qs = User.objects.annotate(n=Count('id'))\n"
        "    for u in qs:\n"
        "        print(u.n)\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 6
    col = src.splitlines()[line].index("u.n") + 2
    diag = _diag(line, col, col + 1, "n")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_annotate_alias_on_get_result_is_dropped(tmp_path: Path):
    """``u = M.objects.annotate(n=…).get(...); u.n`` — terminal queryset
    method binds an instance; alias still resolves through the chain."""
    src = (
        "from django.db.models import Count\n"
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    u = User.objects.annotate(n=Count('id')).get(pk=1)\n"
        "    return u.n\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 5
    col = src.splitlines()[line].index("u.n") + 2
    diag = _diag(line, col, col + 1, "n")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_annotate_alias_across_rebinding_is_dropped(tmp_path: Path):
    """``qs = M.objects.annotate(n=…); qs = qs.filter(...); for u in qs: u.n``
    — alias survives a subsequent rebinding to a filter chain."""
    src = (
        "from django.db.models import Count\n"
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    qs = User.objects.annotate(n=Count('id'))\n"
        "    qs = qs.filter(email__contains='@')\n"
        "    for u in qs:\n"
        "        print(u.n)\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 7
    col = src.splitlines()[line].index("u.n") + 2
    diag = _diag(line, col, col + 1, "n")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_alias_method_works_same_as_annotate(tmp_path: Path):
    """``.alias(x=…)`` declares the same kind of name as ``.annotate(x=…)``."""
    src = (
        "from django.db.models import Count\n"
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    for u in User.objects.alias(n=Count('id')):\n"
        "        print(u.n)\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 5
    col = src.splitlines()[line].index("u.n") + 2
    diag = _diag(line, col, col + 1, "n")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_non_alias_attr_on_annotated_qs_is_kept(tmp_path: Path):
    """A truly unknown attr on an annotated instance is still flagged."""
    src = (
        "from django.db.models import Count\n"
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    for u in User.objects.annotate(n=Count('id')):\n"
        "        print(u.bogus)\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 5
    col = src.splitlines()[line].index("u.bogus") + 2
    diag = _diag(line, col, col + len("bogus"), "bogus")
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_annotate_rule_disabled_keeps_diagnostic(tmp_path: Path):
    from iommi_lsp.config import Config

    src = (
        "from django.db.models import Count\n"
        "from myapp.models import User\n"
        "\n"
        "def f():\n"
        "    for u in User.objects.annotate(n=Count('id')):\n"
        "        print(u.n)\n"
    )
    f = tmp_path / "u.py"
    f.write_text(src)

    a = DjangoAnalyzer(
        workspace_root=CORPUS / "basic_django",
        config=Config(disabled_rules=frozenset({"annotate"})),
    )
    a.django_index = build_index(CORPUS / "basic_django")

    line = 5
    col = src.splitlines()[line].index("u.n") + 2
    diag = _diag(line, col, col + 1, "n")
    assert a.is_false_positive(f.as_uri(), diag) is False


# -- relation field declaration suppression ----------------------------------


def _relation_field_diag(line: int, col_start: int, col_end: int, type_name: str, target: str = "Project") -> dict:
    """Mirror ty's ``invalid-assignment`` for ``f: Target = RelationField(...)``."""
    return {
        "code": "invalid-assignment",
        "message": f"Object of type `{type_name}` is not assignable to `{target}`",
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 1,
        "source": "ty",
    }


@pytest.mark.parametrize(
    "field_type",
    ["ForeignKey", "OneToOneField", "ManyToManyField"],
)
def test_relation_field_assignment_diagnostic_is_dropped(tmp_path: Path, field_type: str):
    src = (
        "from django.db import models\n"
        "\n"
        "class Project(models.Model):\n"
        "    name = models.CharField(max_length=100)\n"
        "\n"
        "class Task(models.Model):\n"
        f"    project: \"Project\" = models.{field_type}(Project, on_delete=models.CASCADE)\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 6
    col_start = src.splitlines()[line].index(f"models.{field_type}")
    col_end = col_start + len(src.splitlines()[line]) - col_start
    diag = _relation_field_diag(line, col_start, col_end, field_type)
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_relation_field_assignment_with_generic_params_is_dropped(tmp_path: Path):
    """django-stubs renders the type as ``ForeignKey[Unknown, Unknown]``."""
    src = (
        "from django.db import models\n"
        "\n"
        "class Project(models.Model):\n"
        "    name = models.CharField(max_length=100)\n"
        "\n"
        "class Task(models.Model):\n"
        "    project: \"Project\" = models.ForeignKey(Project, on_delete=models.CASCADE)\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 6
    line_text = src.splitlines()[line]
    col_start = line_text.index("models.ForeignKey")
    col_end = len(line_text)
    diag = _relation_field_diag(line, col_start, col_end, "ForeignKey[Unknown, Unknown]")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_relation_field_assignment_bare_import_is_dropped(tmp_path: Path):
    """``from django.db.models import ForeignKey`` — bare ``Name`` call."""
    src = (
        "from django.db import models\n"
        "from django.db.models import ForeignKey, CASCADE\n"
        "\n"
        "class Project(models.Model):\n"
        "    name = models.CharField(max_length=100)\n"
        "\n"
        "class Task(models.Model):\n"
        "    project: \"Project\" = ForeignKey(Project, on_delete=CASCADE)\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 7
    line_text = src.splitlines()[line]
    col_start = line_text.index("ForeignKey(")
    col_end = len(line_text)
    diag = _relation_field_diag(line, col_start, col_end, "ForeignKey")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_invalid_assignment_on_non_relation_field_is_kept(tmp_path: Path):
    """Non-relation invalid-assignment on a class body — keep it."""
    src = (
        "class Plain:\n"
        "    x: int = \"oops\"\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 1
    line_text = src.splitlines()[line]
    col_start = line_text.index("\"oops\"")
    col_end = col_start + len("\"oops\"")
    diag = {
        "code": "invalid-assignment",
        "message": "Object of type `str` is not assignable to `int`",
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 1,
        "source": "ty",
    }
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_relation_field_rule_disabled_keeps_diagnostic(tmp_path: Path):
    from iommi_lsp.config import Config

    src = (
        "from django.db import models\n"
        "\n"
        "class Project(models.Model):\n"
        "    name = models.CharField(max_length=100)\n"
        "\n"
        "class Task(models.Model):\n"
        "    project: \"Project\" = models.ForeignKey(Project, on_delete=models.CASCADE)\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(
        workspace_root=CORPUS / "basic_django",
        config=Config(disabled_rules=frozenset({"relation_field_assignment"})),
    )
    a.django_index = build_index(CORPUS / "basic_django")

    line = 6
    line_text = src.splitlines()[line]
    col_start = line_text.index("models.ForeignKey")
    col_end = len(line_text)
    diag = _relation_field_diag(line, col_start, col_end, "ForeignKey")
    assert a.is_false_positive(f.as_uri(), diag) is False


# -- scalar field declaration suppression ------------------------------------


def _scalar_field_diag(line: int, col_start: int, col_end: int, type_name: str, target: str) -> dict:
    """Mirror ty's ``invalid-assignment`` for ``f: target = ScalarField(...)``."""
    return {
        "code": "invalid-assignment",
        "message": f"Object of type `{type_name}` is not assignable to `{target}`",
        "range": {
            "start": {"line": line, "character": col_start},
            "end": {"line": line, "character": col_end},
        },
        "severity": 1,
        "source": "ty",
    }


@pytest.mark.parametrize(
    "field_type, target",
    [
        ("CharField", "str"),
        ("TextField", "str"),
        ("SlugField", "str"),
        ("EmailField", "str"),
        ("IntegerField", "int"),
        ("BigIntegerField", "int"),
        ("PositiveSmallIntegerField", "int"),
        ("AutoField", "int"),
        ("FloatField", "float"),
        ("FloatField", "int | float"),  # PEP 484 numeric tower widening
        ("BooleanField", "bool"),
        ("DecimalField", "Decimal"),
        ("DateField", "date"),
        ("DateTimeField", "datetime"),
        ("DurationField", "timedelta"),
        ("BinaryField", "bytes"),
        ("UUIDField", "UUID"),
        # Nullable declarations — ty spells the target as ``T | None``.
        ("CharField", "str | None"),
        ("IntegerField", "int | None"),
        ("FloatField", "int | float | None"),
    ],
)
def test_scalar_field_assignment_diagnostic_is_dropped(tmp_path: Path, field_type: str, target: str):
    src = (
        "from django.db import models\n"
        "\n"
        "class Thing(models.Model):\n"
        f"    value: object = models.{field_type}()\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    line_text = src.splitlines()[line]
    col_start = line_text.index(f"models.{field_type}")
    col_end = len(line_text)
    diag = _scalar_field_diag(line, col_start, col_end, field_type, target)
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_scalar_field_assignment_with_generic_params_is_dropped(tmp_path: Path):
    """django-stubs renders the type as ``CharField[Unknown, Unknown]``."""
    src = (
        "from django.db import models\n"
        "\n"
        "class Thing(models.Model):\n"
        "    name: str = models.CharField(max_length=100)\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    line_text = src.splitlines()[line]
    col_start = line_text.index("models.CharField")
    col_end = len(line_text)
    diag = _scalar_field_diag(line, col_start, col_end, "CharField[Unknown, Unknown]", "str")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_scalar_field_assignment_bare_import_is_dropped(tmp_path: Path):
    """``from django.db.models import CharField`` — bare ``Name`` call."""
    src = (
        "from django.db import models\n"
        "from django.db.models import CharField\n"
        "\n"
        "class Thing(models.Model):\n"
        "    name: str = CharField(max_length=100)\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 4
    line_text = src.splitlines()[line]
    col_start = line_text.index("CharField(")
    col_end = len(line_text)
    diag = _scalar_field_diag(line, col_start, col_end, "CharField", "str")
    assert a.is_false_positive(f.as_uri(), diag) is True


def test_scalar_field_type_mismatch_is_kept(tmp_path: Path):
    """``count: str = IntegerField()`` — IntegerField yields ``int``, not
    ``str``, so this is a genuine bug and must survive."""
    src = (
        "from django.db import models\n"
        "\n"
        "class Thing(models.Model):\n"
        "    count: str = models.IntegerField()\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    line_text = src.splitlines()[line]
    col_start = line_text.index("models.IntegerField")
    col_end = len(line_text)
    diag = _scalar_field_diag(line, col_start, col_end, "IntegerField", "str")
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_scalar_field_assignment_unknown_field_is_kept(tmp_path: Path):
    """A non-field RHS with the same message shape is not ours."""
    src = (
        "class Plain:\n"
        "    x: int = SomeThing()\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(workspace_root=CORPUS / "basic_django")
    a.django_index = build_index(CORPUS / "basic_django")

    line = 1
    line_text = src.splitlines()[line]
    col_start = line_text.index("SomeThing()")
    col_end = len(line_text)
    diag = _scalar_field_diag(line, col_start, col_end, "SomeThing", "int")
    assert a.is_false_positive(f.as_uri(), diag) is False


def test_scalar_field_rule_disabled_keeps_diagnostic(tmp_path: Path):
    from iommi_lsp.config import Config

    src = (
        "from django.db import models\n"
        "\n"
        "class Thing(models.Model):\n"
        "    name: str = models.CharField(max_length=100)\n"
    )
    f = tmp_path / "m.py"
    f.write_text(src)

    a = DjangoAnalyzer(
        workspace_root=CORPUS / "basic_django",
        config=Config(disabled_rules=frozenset({"scalar_field_assignment"})),
    )
    a.django_index = build_index(CORPUS / "basic_django")

    line = 3
    line_text = src.splitlines()[line]
    col_start = line_text.index("models.CharField")
    col_end = len(line_text)
    diag = _scalar_field_diag(line, col_start, col_end, "CharField", "str")
    assert a.is_false_positive(f.as_uri(), diag) is False
