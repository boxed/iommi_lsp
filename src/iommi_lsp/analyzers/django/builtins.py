"""Static stubs for Django built-in models we want indexed.

These are parsed by the regular AST scrape pipeline (one source string
per Django module, given a synthetic file path under
``<workspace>/.iommi_lsp-builtins/`` so qualname computation works) and
folded into the workspace index. The result: ``User.objects.filter(
email='x')`` resolves even when the user never declares a ``User``
model — the common case where the project uses
``django.contrib.auth.models.User`` directly.

When the workspace defines a class with the same simple name (e.g. a
project's own ``User`` swapping in via ``AUTH_USER_MODEL``), the
workspace wins via :meth:`DjangoIndex.lookup` — see the ``is_builtin``
gate there.

Each stub is just enough field declarations to validate kwargs against
the model. Method definitions, signals, and runtime-only bits are
omitted intentionally; the file isn't executed and isn't a stub in the
``typeshed`` sense.
"""

from __future__ import annotations


CONTRIB_AUTH_MODELS_SRC = """\
from django.db import models


class AbstractBaseUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField()

    class Meta:
        abstract = True


class PermissionsMixin(models.Model):
    is_superuser = models.BooleanField()
    groups = models.ManyToManyField("Group", related_name="user_set")
    user_permissions = models.ManyToManyField(
        "Permission", related_name="user_set"
    )

    class Meta:
        abstract = True


class AbstractUser(AbstractBaseUser, PermissionsMixin):
    username = models.CharField(max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.EmailField()
    is_staff = models.BooleanField()
    is_active = models.BooleanField()
    date_joined = models.DateTimeField()

    class Meta:
        abstract = True


class User(AbstractUser):
    class Meta:
        pass


class Group(models.Model):
    name = models.CharField(max_length=150)
    permissions = models.ManyToManyField("Permission")


class Permission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey("ContentType", on_delete=models.CASCADE)
    codename = models.CharField(max_length=100)
"""


CONTRIB_CONTENTTYPES_MODELS_SRC = """\
from django.db import models


class ContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
"""


CONTRIB_SESSIONS_MODELS_SRC = """\
from django.db import models


class Session(models.Model):
    session_key = models.CharField(max_length=40, primary_key=True)
    session_data = models.TextField()
    expire_date = models.DateTimeField()
"""


# Module qualname -> source. The qualname determines what
# ``Model.qualname`` ends up looking like in the index.
BUILTIN_MODULES: dict[str, str] = {
    "django.contrib.auth.models": CONTRIB_AUTH_MODELS_SRC,
    "django.contrib.contenttypes.models": CONTRIB_CONTENTTYPES_MODELS_SRC,
    "django.contrib.sessions.models": CONTRIB_SESSIONS_MODELS_SRC,
}
